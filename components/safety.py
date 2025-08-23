import streamlit as st
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from config.settings import Settings

logger = logging.getLogger(__name__)

class SafetySystem:
    def __init__(self):
        self.settings = Settings()
        self.safety_rules = self.load_safety_rules()
        self.blocked_patterns = self.load_blocked_patterns()
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\((\d+)\)',  # (1), (2), etc.
            r'reference (\d+)',  # reference 1, reference 2, etc.
        ]
        
    def load_safety_rules(self) -> List[Dict[str, str]]:
        """Load medical safety rules and guidelines"""
        return [
            {
                "rule": "No personalized medical advice",
                "description": "Do not provide specific medical advice for individual patients",
                "keywords": ["should I take", "what medication", "my symptoms", "I have", "my condition"],
                "response": "I cannot provide personalized medical advice. Please consult with a qualified healthcare professional."
            },
            {
                "rule": "No diagnosis",
                "description": "Do not attempt to diagnose medical conditions",
                "keywords": ["do I have", "am I sick", "what's wrong with me", "diagnose", "symptoms mean"],
                "response": "I cannot diagnose medical conditions. Please see a healthcare provider for proper evaluation."
            },
            {
                "rule": "No treatment recommendations",
                "description": "Do not recommend specific treatments without proper medical supervision",
                "keywords": ["should I take", "how much", "stop taking", "treatment for", "cure for"],
                "response": "Treatment decisions should always be made with a qualified healthcare professional."
            },
            {
                "rule": "Emergency situations",
                "description": "Recognize and appropriately respond to medical emergencies",
                "keywords": ["emergency", "urgent", "severe pain", "can't breathe", "chest pain", "suicide"],
                "response": "This appears to be a medical emergency. Please contact emergency services immediately or go to the nearest emergency room."
            },
            {
                "rule": "Medication safety",
                "description": "Emphasize proper medication use and safety",
                "keywords": ["drug interaction", "side effects", "overdose", "mixing medications"],
                "response": "Medication safety is critical. Please consult your pharmacist or healthcare provider about drug interactions and proper usage."
            }
        ]
    
    def load_blocked_patterns(self) -> List[str]:
        """Load patterns that should be blocked in responses"""
        return [
            r"take \d+ pills?",  # Specific dosage instructions
            r"stop taking your medication",  # Dangerous medication advice
            r"you have [a-zA-Z\s]+ disease",  # Diagnosis statements
            r"you should see a doctor in \d+ days",  # Specific medical timing
            r"this will cure",  # Cure claims
            r"guaranteed to work",  # Medical guarantees
            r"FDA approved for",  # False FDA claims
        ]
    
    def get_medical_disclaimer(self) -> str:
        """Get the standard medical disclaimer"""
        return """
        ⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. 
        Always consult with a qualified healthcare provider for medical concerns, diagnosis, or treatment decisions.
        """
    
    def get_safety_system_prompt(self) -> str:
        """Get the system prompt that enforces safety guidelines"""
        return """You are a medical education assistant designed to provide accurate, evidence-based information for learning purposes only. 

CRITICAL SAFETY GUIDELINES:
1. DO NOT provide personalized medical advice, diagnosis, or treatment recommendations
2. DO NOT recommend specific medications or dosages
3. ALWAYS emphasize consulting healthcare professionals for medical decisions
4. Include numbered citations [1], [2] for all medical claims using provided sources
5. If uncertain about medical information, clearly state limitations
6. For emergency situations, direct users to emergency services
7. Focus on general medical education and established medical knowledge

Response format:
- Provide educational information with citations
- Include relevant safety warnings
- End with appropriate disclaimers
- Direct users to healthcare professionals when appropriate"""
    
    def check_input_safety(self, user_input: str) -> Tuple[bool, str, str]:
        """
        Check if user input triggers safety concerns
        Returns: (is_safe, safety_rule_triggered, suggested_response)
        """
        user_input_lower = user_input.lower()
        
        for rule in self.safety_rules:
            for keyword in rule["keywords"]:
                if keyword.lower() in user_input_lower:
                    return False, rule["rule"], rule["response"]
        
        return True, "", ""
    
    def check_output_safety(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check if model output violates safety guidelines
        Returns: (is_safe, list_of_violations)
        """
        violations = []
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(f"Contains blocked pattern: {pattern}")
        
        # Check for lack of citations in medical claims
        medical_claim_patterns = [
            r"studies show",
            r"research indicates",
            r"according to",
            r"evidence suggests",
            r"clinical trials",
            r"proven to",
        ]
        
        has_medical_claims = any(re.search(pattern, response, re.IGNORECASE) for pattern in medical_claim_patterns)
        has_citations = any(re.search(pattern, response) for pattern in self.citation_patterns)
        
        if has_medical_claims and not has_citations:
            violations.append("Medical claims without proper citations")
        
        return len(violations) == 0, violations
    
    def enforce_citation_format(self, response: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """Ensure proper citation format in responses"""
        if not retrieved_contexts:
            return response
        
        # Add citation appendix
        citation_appendix = "\n\n**Sources:**\n"
        for i, context in enumerate(retrieved_contexts, 1):
            source = context.get('source', 'Unknown source')
            citation_appendix += f"[{i}] {source}\n"
        
        return response + citation_appendix
    
    def sanitize_response(self, response: str, retrieved_contexts: List[Dict[str, Any]] = None) -> str:
        """Apply safety measures to model response"""
        # Check output safety
        is_safe, violations = self.check_output_safety(response)
        
        if not is_safe:
            logger.warning(f"Safety violations detected: {violations}")
            return self.get_safe_fallback_response()
        
        # Enforce citations
        if retrieved_contexts:
            response = self.enforce_citation_format(response, retrieved_contexts)
        
        # Add medical disclaimer
        response += "\n\n" + self.get_medical_disclaimer()
        
        return response
    
    def get_safe_fallback_response(self) -> str:
        """Get a safe fallback response when safety violations are detected"""
        return """I understand you're looking for medical information, but I need to be careful about providing safe and appropriate responses. 

For specific medical concerns, symptoms, or treatment questions, I strongly recommend:

1. **Consult a healthcare professional** - They can provide personalized advice based on your specific situation
2. **Contact your doctor or pharmacist** - For medication-related questions
3. **Visit a medical facility** - For urgent concerns
4. **Call emergency services** - For any medical emergency

I'm here to provide general medical education and information, but cannot replace professional medical guidance.

""" + self.get_medical_disclaimer()
    
    def generate_safety_report(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a safety compliance report"""
        total_interactions = len(interactions)
        safety_triggered = 0
        citation_compliance = 0
        violation_types = {}
        
        for interaction in interactions:
            user_input = interaction.get('user_input', '')
            response = interaction.get('response', '')
            
            # Check input safety
            is_input_safe, rule_triggered, _ = self.check_input_safety(user_input)
            if not is_input_safe:
                safety_triggered += 1
                violation_types[rule_triggered] = violation_types.get(rule_triggered, 0) + 1
            
            # Check citation compliance
            has_citations = any(re.search(pattern, response) for pattern in self.citation_patterns)
            if has_citations:
                citation_compliance += 1
        
        return {
            'total_interactions': total_interactions,
            'safety_triggered': safety_triggered,
            'safety_trigger_rate': safety_triggered / total_interactions if total_interactions > 0 else 0,
            'citation_compliance': citation_compliance,
            'citation_compliance_rate': citation_compliance / total_interactions if total_interactions > 0 else 0,
            'violation_breakdown': violation_types,
            'compliance_score': (total_interactions - safety_triggered + citation_compliance) / (2 * total_interactions) if total_interactions > 0 else 1.0
        }
    
    def test_safety_rules(self) -> List[Dict[str, Any]]:
        """Test safety rules with example inputs"""
        test_cases = [
            "Should I take aspirin for my headache?",
            "What medication should I take for high blood pressure?",
            "Do I have cancer based on these symptoms?",
            "How much ibuprofen can I take?",
            "Can you diagnose my rash?",
            "I'm having chest pain, what should I do?",
            "What is the general treatment for diabetes?",  # This should be safe
            "How does blood pressure medication work?",  # This should be safe
        ]
        
        results = []
        for test_input in test_cases:
            is_safe, rule_triggered, suggested_response = self.check_input_safety(test_input)
            results.append({
                'input': test_input,
                'is_safe': is_safe,
                'rule_triggered': rule_triggered,
                'suggested_response': suggested_response
            })
        
        return results
    
    def render(self):
        """Render the safety system interface"""
        st.header("🛡️ Safety & Guardrails System")
        
        # Overview
        st.subheader("Safety Overview")
        st.info("""
        The safety system ensures that the medical AI provides responsible, educational information while avoiding harmful medical advice.
        It includes input filtering, output sanitization, citation enforcement, and compliance monitoring.
        """)
        
        # Safety rules configuration
        st.subheader("Safety Rules Configuration")
        
        # Display current rules
        with st.expander("Current Safety Rules", expanded=True):
            for i, rule in enumerate(self.safety_rules):
                st.write(f"**{i+1}. {rule['rule']}**")
                st.write(f"*Description:* {rule['description']}")
                st.write(f"*Keywords:* {', '.join(rule['keywords'])}")
                st.write(f"*Response:* {rule['response']}")
                st.write("---")
        
        # Add custom safety rule
        st.subheader("Add Custom Safety Rule")
        with st.form("add_safety_rule"):
            new_rule_name = st.text_input("Rule Name:")
            new_rule_desc = st.text_area("Description:")
            new_keywords = st.text_input("Keywords (comma-separated):")
            new_response = st.text_area("Suggested Response:")
            
            if st.form_submit_button("Add Rule"):
                if new_rule_name and new_keywords and new_response:
                    keywords_list = [kw.strip() for kw in new_keywords.split(',')]
                    new_rule = {
                        "rule": new_rule_name,
                        "description": new_rule_desc,
                        "keywords": keywords_list,
                        "response": new_response
                    }
                    self.safety_rules.append(new_rule)
                    st.success("Safety rule added successfully!")
                    st.rerun()
        
        # Safety testing
        st.subheader("Safety Rule Testing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            test_input = st.text_area(
                "Test Input:",
                placeholder="Enter a message to test against safety rules...",
                height=100
            )
            
            if st.button("Test Safety Rules"):
                if test_input:
                    is_safe, rule_triggered, suggested_response = self.check_input_safety(test_input)
                    
                    if is_safe:
                        st.success("✅ Input passed safety checks")
                    else:
                        st.error(f"❌ Safety rule triggered: {rule_triggered}")
                        st.warning(f"Suggested response: {suggested_response}")
        
        with col2:
            if st.button("Run Automated Safety Tests"):
                test_results = self.test_safety_rules()
                
                st.write("**Test Results:**")
                for result in test_results:
                    status = "✅" if result['is_safe'] else "❌"
                    st.write(f"{status} {result['input']}")
                    if not result['is_safe']:
                        st.caption(f"Rule: {result['rule_triggered']}")
        
        # Citation system
        st.subheader("Citation System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Supported Citation Formats:**")
            for pattern in self.citation_patterns:
                st.code(pattern)
        
        with col2:
            citation_test = st.text_area(
                "Test Citation Detection:",
                placeholder="Enter text with citations to test detection...",
                height=100
            )
            
            if citation_test:
                has_citations = any(re.search(pattern, citation_test) for pattern in self.citation_patterns)
                if has_citations:
                    st.success("✅ Citations detected")
                else:
                    st.warning("❌ No citations found")
        
        # Safety monitoring
        st.subheader("Safety Monitoring")
        
        # Generate mock safety report
        if st.button("Generate Safety Report"):
            # This would normally use real interaction data
            mock_interactions = [
                {"user_input": "What is diabetes?", "response": "Diabetes is a condition [1]..."},
                {"user_input": "Should I take aspirin?", "response": "I cannot provide personalized medical advice..."},
                {"user_input": "How does insulin work?", "response": "Insulin helps regulate blood sugar [1]..."},
            ]
            
            report = self.generate_safety_report(mock_interactions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Safety Trigger Rate", f"{report['safety_trigger_rate']:.1%}")
            
            with col2:
                st.metric("Citation Compliance", f"{report['citation_compliance_rate']:.1%}")
            
            with col3:
                st.metric("Overall Compliance Score", f"{report['compliance_score']:.1%}")
            
            if report['violation_breakdown']:
                st.write("**Violation Breakdown:**")
                for violation, count in report['violation_breakdown'].items():
                    st.write(f"- {violation}: {count}")
        
        # System prompts
        st.subheader("System Prompts")
        
        with st.expander("Safety System Prompt"):
            st.text_area(
                "Current Safety System Prompt:",
                value=self.get_safety_system_prompt(),
                height=300,
                disabled=True
            )
        
        with st.expander("Medical Disclaimer"):
            st.text_area(
                "Medical Disclaimer Text:",
                value=self.get_medical_disclaimer(),
                height=100,
                disabled=True
            )
        
        # Export safety configuration
        st.subheader("Export/Import Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Safety Config"):
                config = {
                    "safety_rules": self.safety_rules,
                    "blocked_patterns": self.blocked_patterns,
                    "citation_patterns": self.citation_patterns
                }
                
                config_json = json.dumps(config, indent=2)
                st.download_button(
                    label="Download Safety Configuration",
                    data=config_json,
                    file_name="safety_config.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_config = st.file_uploader(
                "Import Safety Config",
                type=['json'],
                help="Import safety configuration from JSON file"
            )
            
            if uploaded_config and st.button("Import Configuration"):
                try:
                    config = json.load(uploaded_config)
                    self.safety_rules = config.get("safety_rules", self.safety_rules)
                    self.blocked_patterns = config.get("blocked_patterns", self.blocked_patterns)
                    self.citation_patterns = config.get("citation_patterns", self.citation_patterns)
                    st.success("Safety configuration imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing configuration: {str(e)}")
