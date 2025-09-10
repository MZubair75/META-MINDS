# =========================================================
# context_collector.py: Context Collection for Enhanced Question Generation
# =========================================================
# This module collects user context to improve question relevance and quality
# including subject area, analysis objectives, target audience, and dataset background

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
from smart_question_generator import DatasetContext

class ContextCollector:
    """Collects and manages user context for enhanced question generation."""
    
    def __init__(self, context_file: str = "user_context.json", input_folder: str = "input"):
        self.context_file = context_file
        self.input_folder = input_folder
        self.predefined_contexts = self._load_predefined_contexts()
        
    def read_input_folder_context(self) -> Optional[DatasetContext]:
        """Read context from input folder files if they exist and contain data.
        
        Returns:
            DatasetContext if valid context found, None otherwise
        """
        try:
            # Get the project root directory (2 levels up from this script)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            input_folder_path = os.path.join(project_root, self.input_folder)
            
            dataset_bg_path = os.path.join(input_folder_path, "Dataset_Background.txt")
            message_path = os.path.join(input_folder_path, "message.txt")
            
            # Check if input folder and files exist
            logging.info(f"Looking for input folder at: {input_folder_path}")
            if not os.path.exists(input_folder_path):
                logging.info(f"Input folder not found at {input_folder_path}, using interactive context collection")
                return None
            else:
                logging.info(f"✅ Found input folder at: {input_folder_path}")
                
            dataset_background = ""
            senior_message = ""
            
            # Read Dataset_Background.txt
            if os.path.exists(dataset_bg_path):
                with open(dataset_bg_path, 'r', encoding='utf-8') as f:
                    dataset_background = f.read().strip()
                    
            # Read message.txt  
            if os.path.exists(message_path):
                with open(message_path, 'r', encoding='utf-8') as f:
                    senior_message = f.read().strip()
            
            # Parse context from the background file
            if dataset_background and len(dataset_background) > 50:  # Basic validation
                context = self._parse_dataset_background(dataset_background, senior_message)
                if context:
                    logging.info("✅ Successfully loaded context from input folder")
                    return context
                    
            logging.info("Input folder files exist but lack sufficient context data")
            return None
            
        except Exception as e:
            logging.error(f"Error reading input folder context: {e}")
            return None
    
    def _parse_dataset_background(self, background_text: str, message_text: str = "") -> Optional[DatasetContext]:
        """Parse the Dataset_Background.txt content to extract context information.
        
        Args:
            background_text: Content from Dataset_Background.txt
            message_text: Content from message.txt
            
        Returns:
            DatasetContext if parsing successful, None otherwise
        """
        try:
            # Extract key information from the background text
            lines = background_text.lower().split('\n')
            
            # Default values
            subject_area = "general analysis"
            analysis_objectives = []
            target_audience = "analysts"
            business_context = background_text[:500] + "..." if len(background_text) > 500 else background_text
            dataset_background = background_text
            time_sensitivity = "medium"
            
            # Parse subject area and objectives from the text
            for line in lines:
                line = line.strip()
                
                # Look for subject area indicators
                if any(keyword in line for keyword in ["financial", "finance", "asset", "revenue", "profit"]):
                    subject_area = "financial analysis"
                elif any(keyword in line for keyword in ["sales", "revenue", "pipeline", "customer"]):
                    subject_area = "sales performance"
                elif any(keyword in line for keyword in ["marketing", "campaign", "brand", "customer acquisition"]):
                    subject_area = "marketing analytics"
                elif any(keyword in line for keyword in ["operational", "operations", "efficiency", "process"]):
                    subject_area = "operational analytics"
                    
                # Look for analysis objectives
                if "risk" in line and "assessment" in line:
                    analysis_objectives.append("risk assessment")
                if "performance" in line and ("evaluation" in line or "analysis" in line):
                    analysis_objectives.append("performance evaluation")
                if "trend" in line:
                    analysis_objectives.append("trend analysis")
                if "optimization" in line:
                    analysis_objectives.append("optimization analysis")
                    
                # Look for target audience
                if "executive" in line:
                    target_audience = "executives"
                elif "manager" in line:
                    target_audience = "managers"
                elif "analyst" in line:
                    target_audience = "analysts"
                    
                # Look for time sensitivity
                if "urgent" in line or "high" in line:
                    time_sensitivity = "high"
                elif "low" in line:
                    time_sensitivity = "low"
            
            # Ensure we have at least one objective
            if not analysis_objectives:
                analysis_objectives = ["general analysis"]
            
            # Include senior message in business context if provided
            if message_text:
                business_context = f"Senior Instructions: {message_text[:200]}... | Background: {background_text[:300]}..."
            
            context = DatasetContext(
                subject_area=subject_area,
                analysis_objectives=analysis_objectives,
                target_audience=target_audience,
                business_context=business_context,
                dataset_background=dataset_background,
                time_sensitivity=time_sensitivity
            )
            
            logging.info(f"Parsed context - Subject: {subject_area}, Objectives: {analysis_objectives}, Audience: {target_audience}")
            return context
            
        except Exception as e:
            logging.error(f"Error parsing dataset background: {e}")
            return None
    
    def _load_predefined_contexts(self) -> Dict[str, DatasetContext]:
        """Load predefined context templates for common analysis scenarios."""
        return {
            "financial_analysis": DatasetContext(
                subject_area="financial analysis",
                analysis_objectives=["performance evaluation", "risk assessment", "trend analysis", "ROI optimization"],
                target_audience="financial analysts",
                business_context="Investment decisions, portfolio management, and financial planning",
                time_sensitivity="high"
            ),
            "marketing_analytics": DatasetContext(
                subject_area="marketing analytics", 
                analysis_objectives=["campaign effectiveness", "customer segmentation", "ROI analysis", "brand performance"],
                target_audience="marketing managers",
                business_context="Marketing strategy optimization, budget allocation, and customer acquisition",
                time_sensitivity="medium"
            ),
            "operational_analytics": DatasetContext(
                subject_area="operational analytics",
                analysis_objectives=["efficiency optimization", "cost reduction", "process improvement", "quality control"],
                target_audience="operations managers",
                business_context="Operational excellence, resource optimization, and process automation",
                time_sensitivity="high"
            ),
            "sales_analytics": DatasetContext(
                subject_area="sales analytics",
                analysis_objectives=["sales performance", "pipeline analysis", "forecasting", "territory optimization"],
                target_audience="sales managers",
                business_context="Sales strategy, revenue optimization, and performance management",
                time_sensitivity="high"
            ),
            "customer_analytics": DatasetContext(
                subject_area="customer analytics",
                analysis_objectives=["customer behavior", "retention analysis", "satisfaction measurement", "lifetime value"],
                target_audience="customer success managers",
                business_context="Customer experience improvement, retention strategies, and loyalty programs",
                time_sensitivity="medium"
            ),
            "hr_analytics": DatasetContext(
                subject_area="human resources analytics",
                analysis_objectives=["employee performance", "retention analysis", "workforce planning", "diversity metrics"],
                target_audience="HR managers",
                business_context="Talent management, organizational development, and employee engagement",
                time_sensitivity="medium"
            ),
            "supply_chain_analytics": DatasetContext(
                subject_area="supply chain analytics",
                analysis_objectives=["inventory optimization", "demand forecasting", "supplier performance", "logistics efficiency"],
                target_audience="supply chain managers",
                business_context="Supply chain optimization, cost reduction, and risk mitigation",
                time_sensitivity="high"
            ),
            "healthcare_analytics": DatasetContext(
                subject_area="healthcare analytics",
                analysis_objectives=["patient outcomes", "cost analysis", "resource utilization", "quality improvement"],
                target_audience="healthcare administrators",
                business_context="Healthcare delivery optimization, cost management, and patient care improvement",
                time_sensitivity="high"
            ),
            "retail_analytics": DatasetContext(
                subject_area="retail analytics",
                analysis_objectives=["sales optimization", "inventory management", "customer insights", "pricing strategy"],
                target_audience="retail managers",
                business_context="Retail performance optimization, customer experience, and profitability",
                time_sensitivity="medium"
            ),
            "manufacturing_analytics": DatasetContext(
                subject_area="manufacturing analytics",
                analysis_objectives=["production efficiency", "quality control", "equipment maintenance", "cost optimization"],
                target_audience="manufacturing managers",
                business_context="Manufacturing excellence, productivity improvement, and operational efficiency",
                time_sensitivity="high"
            ),
            "energy_analytics": DatasetContext(
                subject_area="energy analytics",
                analysis_objectives=["consumption optimization", "efficiency analysis", "sustainability metrics", "cost reduction"],
                target_audience="energy managers",
                business_context="Energy management, sustainability initiatives, and operational cost optimization",
                time_sensitivity="medium"
            ),
            "cybersecurity_analytics": DatasetContext(
                subject_area="cybersecurity analytics",
                analysis_objectives=["threat detection", "risk assessment", "incident analysis", "security metrics"],
                target_audience="security analysts",
                business_context="Cybersecurity posture improvement, threat mitigation, and risk management",
                time_sensitivity="high"
            ),
            "education_analytics": DatasetContext(
                subject_area="education analytics",
                analysis_objectives=["student performance", "learning outcomes", "resource allocation", "engagement analysis"],
                target_audience="education administrators",
                business_context="Educational excellence, student success, and institutional effectiveness",
                time_sensitivity="medium"
            ),
            "real_estate_analytics": DatasetContext(
                subject_area="real estate analytics",
                analysis_objectives=["market analysis", "price prediction", "investment optimization", "portfolio performance"],
                target_audience="real estate analysts",
                business_context="Real estate investment decisions, market insights, and portfolio optimization",
                time_sensitivity="medium"
            ),
            "transportation_analytics": DatasetContext(
                subject_area="transportation analytics",
                analysis_objectives=["route optimization", "fleet management", "safety analysis", "efficiency improvement"],
                target_audience="transportation managers",
                business_context="Transportation efficiency, cost optimization, and safety improvement",
                time_sensitivity="high"
            ),
            "telecommunications_analytics": DatasetContext(
                subject_area="telecommunications analytics",
                analysis_objectives=["network performance", "customer churn", "service quality", "capacity planning"],
                target_audience="telecom analysts",
                business_context="Network optimization, customer retention, and service quality improvement",
                time_sensitivity="medium"
            )
        }
    
    def collect_context_hybrid(self) -> DatasetContext:
        """Hybrid context collection - tries input folder first, falls back to interactive.
        
        Returns:
            DatasetContext with complete business context
        """
        # Try to read from input folder first
        folder_context = self.read_input_folder_context()
        
        if folder_context:
            print("✅ Using business context from input folder")
            print(f"   📊 Subject Area: {folder_context.subject_area}")
            print(f"   🎯 Objectives: {', '.join(folder_context.analysis_objectives)}")
            print(f"   👥 Audience: {folder_context.target_audience}")
            print()
            return folder_context
        else:
            print("📝 Input folder context not available, collecting interactively...")
            return self.collect_context_interactive()
    
    def collect_context_interactive(self) -> DatasetContext:
        """Collect context through interactive user prompts."""
        print("\n" + "="*60)
        print("📊 META MINDS - ENHANCED CONTEXT COLLECTION")
        print("="*60)
        print("To generate the most relevant analytical questions, please provide some context:")
        print()
        
        # Option for predefined contexts
        use_predefined = self._get_predefined_context_choice()
        if use_predefined:
            return use_predefined
        
        # Collect custom context
        context = DatasetContext()
        
        # Subject Area
        context.subject_area = self._get_subject_area()
        
        # Analysis Objectives  
        context.analysis_objectives = self._get_analysis_objectives()
        
        # Target Audience
        context.target_audience = self._get_target_audience()
        
        # Dataset Background
        context.dataset_background = self._get_dataset_background()
        
        # Business Context
        context.business_context = self._get_business_context()
        
        # Time Sensitivity
        context.time_sensitivity = self._get_time_sensitivity()
        
        # Save context for future use
        self._save_context(context)
        
        print("\n✅ Context collection complete!")
        print(f"📋 Analysis Focus: {context.subject_area}")
        print(f"🎯 Primary Objectives: {', '.join(context.analysis_objectives[:2])}...")
        print(f"👥 Target Audience: {context.target_audience}")
        print()
        
        return context
    
    def _get_predefined_context_choice(self) -> Optional[DatasetContext]:
        """Allow user to choose from predefined contexts."""
        print("Would you like to use a predefined context template? (recommended for faster setup)")
        print()
        
        contexts = list(self.predefined_contexts.keys())
        for i, context_name in enumerate(contexts, 1):
            context = self.predefined_contexts[context_name]
            print(f"{i}. {context.subject_area.title()}")
            print(f"   Focus: {', '.join(context.analysis_objectives[:2])}")
            print(f"   Audience: {context.target_audience}")
            print()
        
        print(f"{len(contexts) + 1}. Custom context (I'll provide my own details)")
        print()
        
        while True:
            try:
                choice = input("Enter your choice (1-{}) or press Enter for custom: ".format(len(contexts) + 1)).strip()
                
                if not choice:  # Enter pressed - use custom
                    return None
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(contexts):
                    selected_context = self.predefined_contexts[contexts[choice_num - 1]]
                    print(f"\n✅ Selected: {selected_context.subject_area.title()}")
                    
                    # Allow minor customizations
                    if self._confirm_customization():
                        return self._customize_predefined_context(selected_context)
                    else:
                        return selected_context
                        
                elif choice_num == len(contexts) + 1:
                    return None  # Use custom context
                else:
                    print("❌ Invalid choice. Please select a valid option.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def _confirm_customization(self) -> bool:
        """Ask if user wants to customize the predefined context."""
        while True:
            customize = input("Would you like to customize this template? (y/n): ").strip().lower()
            if customize in ['y', 'yes']:
                return True
            elif customize in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def _customize_predefined_context(self, base_context: DatasetContext) -> DatasetContext:
        """Allow user to customize a predefined context."""
        print(f"\nCustomizing {base_context.subject_area} template:")
        print()
        
        # Create a copy to modify
        context = DatasetContext(
            subject_area=base_context.subject_area,
            analysis_objectives=base_context.analysis_objectives.copy(),
            target_audience=base_context.target_audience,
            dataset_background=base_context.dataset_background,
            business_context=base_context.business_context,
            time_sensitivity=base_context.time_sensitivity
        )
        
        # Allow modifications
        new_objectives = input(f"Analysis objectives (current: {', '.join(context.analysis_objectives)})\nEnter new objectives (comma-separated) or press Enter to keep current: ").strip()
        if new_objectives:
            context.analysis_objectives = [obj.strip() for obj in new_objectives.split(',')]
        
        new_audience = input(f"Target audience (current: {context.target_audience})\nEnter new audience or press Enter to keep current: ").strip()
        if new_audience:
            context.target_audience = new_audience
            
        new_business_context = input(f"Business context (current: {context.business_context})\nEnter new context or press Enter to keep current: ").strip()
        if new_business_context:
            context.business_context = new_business_context
            
        return context
    
    def _get_subject_area(self) -> str:
        """Get the subject area for analysis."""
        print("📋 SUBJECT AREA")
        print("What domain or field does your data relate to?")
        print("Examples: financial analysis, marketing analytics, sales performance, customer behavior, etc.")
        print()
        
        while True:
            subject_area = input("Subject area: ").strip()
            if subject_area:
                return subject_area.lower()
            print("❌ Please provide a subject area.")
    
    def _get_analysis_objectives(self) -> List[str]:
        """Get the analysis objectives."""
        print("\n🎯 ANALYSIS OBJECTIVES")
        print("What are your main goals for analyzing this data? (separate multiple objectives with commas)")
        print("Examples: trend analysis, performance evaluation, risk assessment, forecasting, etc.")
        print()
        
        while True:
            objectives_input = input("Analysis objectives: ").strip()
            if objectives_input:
                objectives = [obj.strip().lower() for obj in objectives_input.split(',')]
                return [obj for obj in objectives if obj]  # Remove empty strings
            print("❌ Please provide at least one analysis objective.")
    
    def _get_target_audience(self) -> str:
        """Get the target audience for the analysis."""
        print("\n👥 TARGET AUDIENCE")
        print("Who will be using these analytical insights?")
        print("Examples: executives, data analysts, marketing managers, financial analysts, etc.")
        print()
        
        while True:
            audience = input("Target audience: ").strip()
            if audience:
                return audience.lower()
            print("❌ Please specify the target audience.")
    
    def _get_dataset_background(self) -> str:
        """Get background information about the dataset."""
        print("\n📊 DATASET BACKGROUND")
        print("Please provide some background about your dataset(s):")
        print("Examples: source of data, time period covered, data collection method, etc.")
        print("(Optional - press Enter to skip)")
        print()
        
        background = input("Dataset background: ").strip()
        return background if background else "No background information provided"
    
    def _get_business_context(self) -> str:
        """Get business context for the analysis."""
        print("\n💼 BUSINESS CONTEXT")
        print("What business decisions or strategies will this analysis support?")
        print("Examples: budget allocation, strategic planning, process optimization, etc.")
        print("(Optional - press Enter to skip)")
        print()
        
        context = input("Business context: ").strip()
        return context if context else "General business analysis"
    
    def _get_time_sensitivity(self) -> str:
        """Get the time sensitivity of the analysis."""
        print("\n⏰ TIME SENSITIVITY")
        print("How time-sensitive is this analysis?")
        print("1. High (urgent, immediate decisions)")
        print("2. Medium (important, near-term planning)")
        print("3. Low (exploratory, long-term insights)")
        print()
        
        while True:
            try:
                choice = input("Time sensitivity (1-3): ").strip()
                if choice == '1':
                    return "high"
                elif choice == '2':
                    return "medium"
                elif choice == '3':
                    return "low"
                else:
                    print("❌ Please enter 1, 2, or 3.")
            except:
                print("❌ Please enter a valid choice.")
    
    def _save_context(self, context: DatasetContext) -> None:
        """Save context to file for future reference."""
        try:
            context_data = asdict(context)
            context_data['timestamp'] = datetime.now().isoformat()
            
            # Load existing contexts if file exists
            existing_contexts = []
            if os.path.exists(self.context_file):
                try:
                    with open(self.context_file, 'r', encoding='utf-8') as f:
                        existing_contexts = json.load(f)
                except:
                    existing_contexts = []
            
            # Add new context
            existing_contexts.append(context_data)
            
            # Keep only last 10 contexts
            existing_contexts = existing_contexts[-10:]
            
            # Save updated contexts
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(existing_contexts, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Context saved to {self.context_file}")
            
        except Exception as e:
            logging.warning(f"Could not save context: {e}")
    
    def load_recent_context(self) -> Optional[DatasetContext]:
        """Load the most recent context from saved contexts."""
        try:
            if not os.path.exists(self.context_file):
                return None
                
            with open(self.context_file, 'r', encoding='utf-8') as f:
                contexts = json.load(f)
                
            if not contexts:
                return None
                
            # Get most recent context
            recent_context_data = contexts[-1]
            
            # Remove timestamp for DatasetContext creation
            recent_context_data.pop('timestamp', None)
            
            return DatasetContext(**recent_context_data)
            
        except Exception as e:
            logging.warning(f"Could not load recent context: {e}")
            return None
    
    def get_quick_context(self, dataset_names: List[str]) -> DatasetContext:
        """Get a quick context based on dataset names and minimal input."""
        print("\n⚡ QUICK CONTEXT SETUP")
        print("For faster processing, we'll infer context from your dataset names:")
        print(f"Datasets: {', '.join(dataset_names)}")
        print()
        
        # Try to infer subject area from dataset names
        inferred_subject = self._infer_subject_area(dataset_names)
        
        print(f"Inferred subject area: {inferred_subject}")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            # Use predefined context if available
            for key, context in self.predefined_contexts.items():
                if inferred_subject in context.subject_area:
                    print(f"Using {context.subject_area} template")
                    return context
        
        # Fallback to minimal custom context
        subject_area = input("Enter subject area: ").strip() or inferred_subject
        objectives = input("Main objective (e.g., trend analysis): ").strip() or "exploratory analysis"
        
        return DatasetContext(
            subject_area=subject_area,
            analysis_objectives=[objectives],
            target_audience="data analysts",
            dataset_background=f"Analysis of {', '.join(dataset_names)}",
            business_context="Data-driven insights and decision support"
        )
    
    def _infer_subject_area(self, dataset_names: List[str]) -> str:
        """Infer subject area from dataset names."""
        combined_names = ' '.join(dataset_names).lower()
        
        if any(word in combined_names for word in ['stock', 'price', 'financial', 'revenue', 'profit']):
            return "financial analysis"
        elif any(word in combined_names for word in ['sales', 'customer', 'marketing', 'campaign']):
            return "sales and marketing analytics"
        elif any(word in combined_names for word in ['employee', 'hr', 'payroll', 'performance']):
            return "human resources analytics"
        elif any(word in combined_names for word in ['inventory', 'supply', 'logistics', 'operations']):
            return "operational analytics"
        else:
            return "general data analytics"

# Import pandas here to avoid circular imports
import pandas as pd
