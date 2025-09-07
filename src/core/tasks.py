from crewai import Task
import pandas as pd # Needed for DataFrame type hinting
import logging # Although not strictly needed inside tasks, useful for logging task creation
from smart_question_generator import SMARTQuestionGenerator, DatasetContext, create_smart_comparison_questions
from smart_validator import SMARTValidator
from context_collector import ContextCollector

def create_tasks(datasets: list[tuple[str, pd.DataFrame]], agent1, agent2, context: DatasetContext = None, individual_question_count: int = 20) -> tuple[list[Task], list[str]]:
    """Creates individual tasks for the AI agents to generate questions for each dataset.

    Args:
        datasets (list): A list of tuples containing the dataset name (str) and DataFrame (pd.DataFrame).
        agent1 (crewai.Agent): The schema sleuth agent (though not used in the *current* task definition,
                                included for completeness if schema tasks were added).
        agent2 (crewai.Agent): The question genius agent.

    Returns:
        tuple: A tuple containing the list of generated crewai.Task objects
               and the list of corresponding header strings for output.
    """
    logging.info(f"Creating individual dataset analysis tasks for {len(datasets)} dataset(s)...")
    tasks = []
    headers = []

    # Note: The original code had both agents as arguments but only used agent2 (question_genius)
    # for the question generation task. The schema analysis/description was done outside CrewAI.
    # If you wanted a CrewAI task for schema analysis, you would define it here as well,
    # potentially using agent1 (schema_sleuth). For now, adhering to the original task structure.

    for name, df in datasets:
        # Limit sample to avoid exceeding token limits for very wide dataframes
        sample_string = df.head().to_string()
        if len(sample_string) > 2000: # Arbitrary limit, adjust as needed
             sample_string = df.head().to_string()[:2000] + "\n[...truncated...]"
             logging.warning(f"Sample for task description for '{name}' was truncated.")


        question_task = Task(
            description=f"""You are given a single dataset named '{name}'. Your goal is to generate insightful questions for data analysis.
            
**Constraints:**
1. **STRICTLY use ONLY the data provided in this dataset ('{name}').**
2. Do NOT reference or compare with any other dataset, file, or external information.
3. Generate exactly {individual_question_count} distinct, meaningful, and diverse analytical questions.
4. Questions should focus on identifying trends, relationships, anomalies, potential KPIs, or areas for deeper investigation *within* this dataset.
5. Ensure questions are clear and actionable for a data analyst.

Here is a sample from the dataset to help you understand its content:

{sample_string}""",
            agent=agent2, # Assign this task to the question_genius agent
            expected_output=f"""A numbered list (1. 2. etc.) of exactly {individual_question_count} analytical questions based *only* on the '{name}' dataset.
            Start your output with the exact string: "--- Questions for {name} ---" """,
            human_input=False # Typically tasks don't need human input in this flow
        )
        tasks.append(question_task)
        headers.append(f"--- Questions for {name} ---")

    logging.info(f"Created {len(tasks)} individual dataset analysis tasks.")
    return tasks, headers


def create_comparison_task(datasets: list[tuple[str, pd.DataFrame]], agent, comparison_question_count: int = 15) -> Task | None:
    """Creates a task for the AI agent to generate comparison questions across multiple datasets.

    This task is only created if there is more than one dataset provided.

    Args:
        datasets (list): A list of tuples containing the dataset name (str) and DataFrame (pd.DataFrame).
        agent (crewai.Agent): The question genius agent.

    Returns:
        crewai.Task | None: The comparison task if more than one dataset exists, otherwise None.
    """
    if len(datasets) <= 1:
        logging.info("Only one dataset provided, skipping comparison task creation.")
        return None # No comparison needed for a single file
    
    if comparison_question_count <= 0:
        logging.info("Comparison question count is 0, skipping comparison task creation.")
        return None # User chose to skip comparison

    logging.info(f"Creating comparison analysis task for {len(datasets)} datasets with {comparison_question_count} questions...")

    # Concatenate samples from all datasets for the comparison prompt
    comparison_sample_string = ""
    for name, df in datasets:
         sample_string = df.head().to_string()
         if len(sample_string) > 1000: # Truncate samples for comparison prompt
             sample_string = df.head().to_string()[:1000] + "\n[...truncated...]"
             logging.warning(f"Sample for comparison task description for '{name}' was truncated.")
         comparison_sample_string += f"\nDataset '{name}':\n{sample_string}\n"
         comparison_sample_string += "-"*20 + "\n" # Separator

    comparison_task = Task(
        description=f"""You are given multiple datasets with the goal of generating questions that compare and contrast them.

**Constraints:**
1. Generate exactly {comparison_question_count} meaningful and diverse analytical questions.
2. Questions MUST focus on identifying trends, differences, similarities, or potential insights that can be drawn *specifically* by comparing and contrasting the provided datasets.
3. Do NOT generate questions that are specific to only one dataset; focus on comparative analysis.
4. Do NOT reference any external data or knowledge.
5. Ensure questions are clear and actionable for a data analyst performing a comparative study.

Here are samples from the datasets:

{comparison_sample_string}""",
        agent=agent, # Assign this task to the question_genius agent
        expected_output=f"""A numbered list (1. 2. etc.) of {comparison_question_count} analytical questions that compare and contrast the provided datasets.
        Start your output with the exact string: "--- Comparison Questions ---" """,
        human_input=False # Typically tasks don't need human input in this flow
    )
    logging.info("Comparison analysis task created.")
    return comparison_task


def create_smart_tasks(datasets: list[tuple[str, pd.DataFrame]], agent1, agent2, 
                      context: DatasetContext = None, individual_question_count: int = 20) -> tuple[list[Task], list[str], dict]:
    """Creates SMART-enhanced tasks with advanced question generation and validation.
    
    Args:
        datasets: List of (dataset_name, dataframe) tuples
        agent1: Schema sleuth agent  
        agent2: Question genius agent
        context: Enhanced context for question generation
        
    Returns:
        tuple: (tasks, headers, quality_report)
    """
    logging.info(f"Creating SMART-enhanced tasks for {len(datasets)} dataset(s)...")
    
    # Initialize SMART components
    smart_generator = SMARTQuestionGenerator()
    validator = SMARTValidator()
    
    # Use default context if none provided
    if context is None:
        context = DatasetContext()
        logging.warning("No context provided, using default context")
    
    tasks = []
    headers = []
    quality_reports = {}
    
    # Generate enhanced questions for each dataset
    for name, df in datasets:
        logging.info(f"Generating SMART questions for {name}...")
        
        # Generate SMART-compliant questions
        smart_questions = smart_generator.generate_enhanced_questions(
            dataset_name=name,
            df=df,
            context=context,
            num_questions=individual_question_count
        )
        
        # Validate question quality
        validation_report = validator.validate_question_set(smart_questions, context)
        quality_reports[name] = validation_report
        
        # Extract top questions for CrewAI task
        top_questions = [q['question'] for q in smart_questions]
        questions_text = '\n'.join(f"{i+1}. {q}" for i, q in enumerate(top_questions))
        
        # Create enhanced task description
        enhanced_description = f"""You are analyzing the dataset '{name}' with the following context:
        
**Analysis Context:**
- Subject Area: {context.subject_area}
- Analysis Objectives: {', '.join(context.analysis_objectives)}
- Target Audience: {context.target_audience}
- Business Context: {context.business_context}

**Dataset Overview:**
- Rows: {len(df)}, Columns: {len(df.columns)}
- Sample Data:
{df.head(3).to_string()}

**Your Task:**
Review and refine the following SMART-compliant analytical questions. Ensure each question is:
- **Specific**: Targets distinct variables or trends in the data
- **Measurable**: Refers to quantifiable outcomes or metrics  
- **Action-Oriented**: Uses analytical verbs that prompt investigation
- **Relevant**: Relates to the business context and stakeholder interests
- **Time-Bound**: References periods or changes over time where applicable

**Generated Questions to Review:**
{questions_text}

**Instructions:**
1. Review each question for SMART compliance
2. Improve any questions that could be more specific, measurable, or actionable
3. Ensure questions align with the stated analysis objectives
4. Output exactly {individual_question_count} refined, high-quality analytical questions
5. Start each question with "What", "How", or "Why" for open-ended exploration
"""

        # Create the task
        smart_task = Task(
            description=enhanced_description,
            agent=agent2,
            expected_output=f"""A numbered list of exactly {individual_question_count} refined, SMART-compliant analytical questions for {name}.
            Start your output with: "--- Enhanced Questions for {name} ---"
            
Each question must be:
- Specific to variables in the dataset
- Measurable in outcomes
- Action-oriented for analysis
- Relevant to {context.subject_area}
- Time-bound where applicable""",
            human_input=False
        )
        
        tasks.append(smart_task)
        headers.append(f"--- Enhanced Questions for {name} ---")
    
    logging.info(f"Created {len(tasks)} SMART-enhanced tasks")
    
    # Log quality summary
    overall_scores = []
    for name, report in quality_reports.items():
        avg_score = report['summary']['average_score']
        overall_scores.append(avg_score)
        logging.info(f"Average question quality for {name}: {avg_score:.2f}")
    
    if overall_scores:
        logging.info(f"Overall average question quality: {sum(overall_scores)/len(overall_scores):.2f}")
    
    return tasks, headers, quality_reports


def create_smart_comparison_task(datasets: list[tuple[str, pd.DataFrame]], agent, 
                               context: DatasetContext = None, comparison_question_count: int = 15) -> tuple[Task, dict]:
    """Creates SMART-enhanced comparison task for multiple datasets.
    
    Args:
        datasets: List of (dataset_name, dataframe) tuples
        agent: Question genius agent
        context: Enhanced context for question generation
        
    Returns:
        tuple: (comparison_task, quality_report)
    """
    if len(datasets) <= 1:
        logging.info("Only one dataset provided, skipping SMART comparison task creation.")
        return None, {}
    
    if comparison_question_count <= 0:
        logging.info("Comparison question count is 0, skipping SMART comparison task creation.")
        return None, {}

    logging.info(f"Creating SMART comparison analysis task for {len(datasets)} datasets with {comparison_question_count} questions...")
    
    # Use default context if none provided
    if context is None:
        context = DatasetContext()
    
    # Generate SMART comparison questions
    smart_questions = create_smart_comparison_questions(datasets, context, num_questions=comparison_question_count)
    
    # Validate comparison questions
    validator = SMARTValidator()
    validation_report = validator.validate_question_set(smart_questions, context)
    
    # Extract questions for task
    questions_text = '\n'.join(f"{i+1}. {q['question']}" for i, q in enumerate(smart_questions))
    
    # Dataset information for context
    dataset_info = ""
    for name, df in datasets:
        dataset_info += f"\n- {name}: {len(df)} rows, {len(df.columns)} columns"
        dataset_info += f"\n  Sample: {df.head(2).to_string()[:200]}...\n"
    
    enhanced_description = f"""You are conducting a comparative analysis across multiple datasets with this context:

**Analysis Context:**
- Subject Area: {context.subject_area}
- Analysis Objectives: {', '.join(context.analysis_objectives)}
- Target Audience: {context.target_audience}
- Business Context: {context.business_context}

**Datasets for Comparison:**
{dataset_info}

**Your Task:**
Review and refine the following SMART-compliant comparative questions. Each question should:
- **Specific**: Target distinct cross-dataset comparisons
- **Measurable**: Reference quantifiable differences or similarities
- **Action-Oriented**: Use comparative analytical verbs
- **Relevant**: Focus on insights that drive business decisions
- **Time-Bound**: Include temporal comparisons where applicable

**Generated Comparison Questions to Review:**
{questions_text}

**Instructions:**
1. Refine each question for maximum SMART compliance
2. Ensure questions require analysis of multiple datasets together
3. Focus on actionable comparative insights
4. Output exactly {comparison_question_count} enhanced comparative questions
5. Avoid questions specific to only one dataset
"""

    comparison_task = Task(
        description=enhanced_description,
        agent=agent,
        expected_output=f"""A numbered list of exactly {comparison_question_count} refined, SMART-compliant comparative questions.
        Start your output with: "--- Enhanced Comparison Questions ---"
        
Each question must:
- Compare specific elements across datasets
- Reference measurable differences or patterns
- Be action-oriented for comparative analysis
- Drive relevant business insights
- Include temporal context where applicable""",
        human_input=False
    )
    
    avg_score = validation_report['summary']['average_score']
    logging.info(f"SMART comparison task created with average question quality: {avg_score:.2f}")
    
    return comparison_task, validation_report