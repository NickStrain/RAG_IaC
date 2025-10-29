import ollama 
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv  
load_dotenv()


class TerraformValidationAgent:
    """
    Advanced validation agent that checks syntax, structure, and correctness
    """
    def __init__(self, generation_model="codellama:7b-instruct"):
        self.generation_model = generation_model
        self.max_retry_attempts = 3
        
    def validate_hcl_syntax(self, code):
        """
        Check for basic HCL syntax errors
        """
        errors = []
        warnings = []
        
        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} opening vs {close_braces} closing")
        
        # Check for balanced quotes
        double_quotes = code.count('"')
        if double_quotes % 2 != 0:
            errors.append("Unbalanced double quotes")
        
        # Check for balanced brackets
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        if open_brackets != close_brackets:
            errors.append(f"Unbalanced brackets: {open_brackets} opening vs {close_brackets} closing")
        
        # Check for balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} opening vs {close_parens} closing")
        
        # Check for common syntax mistakes
        if re.search(r'=\s*=', code):
            errors.append("Double equals '==' found (use single '=' for assignments)")
        
        if re.search(r'\$\{[^}]*[^}]$', code, re.MULTILINE):
            warnings.append("Possible unclosed interpolation syntax")
        
        # Check for required equals sign in assignments
        if re.search(r'^\s*\w+\s+{', code, re.MULTILINE):
            warnings.append("Possible missing '=' in block definition")
        
        return {
            'syntax_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_terraform_structure(self, terraform_files):
        """
        Validate Terraform file structure and required components
        """
        if isinstance(terraform_files, str):
            try:
                json_match = re.search(r'\{.*\}', terraform_files, re.DOTALL)
                if json_match:
                    terraform_files = json.loads(json_match.group())
                else:
                    terraform_files = {'main_tf': terraform_files}
            except:
                terraform_files = {'main_tf': terraform_files}
        
        main_content = terraform_files.get('main_tf', '')
        variables_content = terraform_files.get('variables_tf', '')
        outputs_content = terraform_files.get('outputs_tf', '')
        
        issues = []
        score = 0
        max_score = 10
        
        # Check terraform block (required)
        if 'terraform {' in main_content or 'terraform{' in main_content:
            score += 2
        else:
            issues.append("CRITICAL: Missing 'terraform {}' block in main.tf")
        
        # Check provider block (required)
        if re.search(r'provider\s+"[\w-]+"', main_content):
            score += 2
        else:
            issues.append("CRITICAL: Missing 'provider' block in main.tf")
        
        # Check resource blocks (required)
        resource_count = len(re.findall(r'resource\s+"[\w-]+"\s+"[\w-]+"', main_content))
        if resource_count > 0:
            score += 2
        else:
            issues.append("CRITICAL: No 'resource' blocks found in main.tf")
        
        # Check variable declarations
        if variables_content:
            var_count = len(re.findall(r'variable\s+"[\w-]+"', variables_content))
            if var_count > 0:
                score += 1
            else:
                issues.append("WARNING: variables.tf exists but has no variable declarations")
        else:
            issues.append("INFO: No variables.tf file generated")
        
        # Check output blocks
        if outputs_content:
            output_count = len(re.findall(r'output\s+"[\w-]+"', outputs_content))
            if output_count > 0:
                score += 1
            else:
                issues.append("WARNING: outputs.tf exists but has no output declarations")
        
        # Check for required_providers
        if 'required_providers' in main_content:
            score += 1
        else:
            issues.append("WARNING: Missing 'required_providers' block")
        
        # Check for provider version constraints
        if re.search(r'version\s*=\s*["\']', main_content):
            score += 1
        else:
            issues.append("WARNING: No version constraints specified for providers")
        
        return {
            'structure_valid': score >= 6,  # At least 60% score required
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'issues': issues,
            'resource_count': resource_count
        }
    
    def validate_terraform_best_practices(self, terraform_files):
        """
        Check for Terraform best practices
        """
        if isinstance(terraform_files, str):
            try:
                json_match = re.search(r'\{.*\}', terraform_files, re.DOTALL)
                if json_match:
                    terraform_files = json.loads(json_match.group())
                else:
                    terraform_files = {'main_tf': terraform_files}
            except:
                terraform_files = {'main_tf': terraform_files}
        
        main_content = terraform_files.get('main_tf', '')
        suggestions = []
        
        # Check for hardcoded values
        if re.search(r'=\s*"[^${"]*"', main_content):
            if not re.search(r'var\.', main_content):
                suggestions.append("Consider using variables instead of hardcoded values")
        
        # Check for tags
        if 'tags' not in main_content and 'resource "aws_' in main_content:
            suggestions.append("Consider adding tags to AWS resources for better organization")
        
        # Check for descriptions in variables
        variables_content = terraform_files.get('variables_tf', '')
        if variables_content and 'variable' in variables_content:
            if 'description' not in variables_content:
                suggestions.append("Add descriptions to variables for better documentation")
        
        # Check for outputs
        if not terraform_files.get('outputs_tf'):
            suggestions.append("Consider adding outputs to expose important resource attributes")
        
        # Check for remote state
        if 'backend' not in main_content:
            suggestions.append("Consider configuring remote state backend for team collaboration")
        
        # Check for data sources
        has_data_sources = 'data "' in main_content
        
        return {
            'suggestions': suggestions,
            'has_data_sources': has_data_sources,
            'best_practices_score': max(0, 10 - len(suggestions))
        }
    
    def llm_semantic_validation(self, terraform_code, original_query, user_variables):
        """
        Use LLM to validate if the generated code matches the requirements
        """
        prompt = f"""You are a Terraform expert validator. Analyze the generated Terraform code and determine if it correctly implements the user's requirements.

Original User Request: {original_query}

User Variables: {json.dumps(user_variables, indent=2)}

Generated Terraform Code:
{terraform_code}

Analyze the code and respond in this EXACT JSON format:
{{
    "matches_requirements": true/false,
    "correctness_score": 0-10,
    "issues_found": ["list of issues if any"],
    "missing_components": ["list of missing components if any"],
    "incorrect_configurations": ["list of incorrect configs if any"],
    "recommendation": "fix/approve/regenerate"
}}

Be strict in your validation. Only approve if the code fully implements the requirements.
"""
        
        response = ollama.generate(model=self.generation_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                validation_result = json.loads(json_match.group())
            else:
                validation_result = {
                    "matches_requirements": False,
                    "correctness_score": 5,
                    "issues_found": ["Could not parse validation response"],
                    "missing_components": [],
                    "incorrect_configurations": [],
                    "recommendation": "regenerate"
                }
        except:
            validation_result = {
                "matches_requirements": False,
                "correctness_score": 5,
                "issues_found": ["Validation parsing failed"],
                "missing_components": [],
                "incorrect_configurations": [],
                "recommendation": "regenerate"
            }
        
        return validation_result
    
    def comprehensive_validation(self, terraform_code, original_query, user_variables):
        """
        Run all validation checks
        """
        print("\n" + "="*70)
        print("🔍 COMPREHENSIVE VALIDATION AGENT")
        print("="*70 + "\n")
        
        # Parse terraform files
        if isinstance(terraform_code, str):
            try:
                json_match = re.search(r'\{.*\}', terraform_code, re.DOTALL)
                if json_match:
                    terraform_files = json.loads(json_match.group())
                else:
                    terraform_files = {'main_tf': terraform_code}
            except:
                terraform_files = {'main_tf': terraform_code}
        else:
            terraform_files = terraform_code
        
        # 1. Syntax Validation
        print("📋 Step 1: Syntax Validation")
        main_syntax = self.validate_hcl_syntax(terraform_files.get('main_tf', ''))
        vars_syntax = self.validate_hcl_syntax(terraform_files.get('variables_tf', ''))
        outputs_syntax = self.validate_hcl_syntax(terraform_files.get('outputs_tf', ''))
        
        syntax_valid = main_syntax['syntax_valid'] and vars_syntax['syntax_valid'] and outputs_syntax['syntax_valid']
        
        all_errors = main_syntax['errors'] + vars_syntax['errors'] + outputs_syntax['errors']
        all_warnings = main_syntax['warnings'] + vars_syntax['warnings'] + outputs_syntax['warnings']
        
        if syntax_valid:
            print("  ✅ Syntax validation PASSED")
        else:
            print("  ❌ Syntax validation FAILED")
            for error in all_errors:
                print(f"     - {error}")
        
        if all_warnings:
            print("  ⚠️  Warnings:")
            for warning in all_warnings:
                print(f"     - {warning}")
        print()
        
        # 2. Structure Validation
        print("📋 Step 2: Structure Validation")
        structure_result = self.validate_terraform_structure(terraform_files)
        
        if structure_result['structure_valid']:
            print(f"  ✅ Structure validation PASSED ({structure_result['percentage']:.1f}%)")
        else:
            print(f"  ❌ Structure validation FAILED ({structure_result['percentage']:.1f}%)")
        
        for issue in structure_result['issues']:
            if 'CRITICAL' in issue:
                print(f"     ❌ {issue}")
            elif 'WARNING' in issue:
                print(f"     ⚠️  {issue}")
            else:
                print(f"     ℹ️  {issue}")
        print()
        
        # 3. Best Practices Check
        print("📋 Step 3: Best Practices Check")
        best_practices = self.validate_terraform_best_practices(terraform_files)
        print(f"  Score: {best_practices['best_practices_score']}/10")
        if best_practices['suggestions']:
            print("  💡 Suggestions:")
            for suggestion in best_practices['suggestions']:
                print(f"     - {suggestion}")
        else:
            print("  ✅ All best practices followed")
        print()
        
        # 4. Semantic Validation
        print("📋 Step 4: Semantic/Requirements Validation")
        semantic_result = self.llm_semantic_validation(
            json.dumps(terraform_files, indent=2),
            original_query,
            user_variables
        )
        
        if semantic_result['matches_requirements']:
            print(f"  ✅ Requirements validation PASSED (Score: {semantic_result['correctness_score']}/10)")
        else:
            print(f"  ❌ Requirements validation FAILED (Score: {semantic_result['correctness_score']}/10)")
        
        if semantic_result['issues_found']:
            print("  Issues:")
            for issue in semantic_result['issues_found']:
                print(f"     - {issue}")
        
        if semantic_result['missing_components']:
            print("  Missing:")
            for missing in semantic_result['missing_components']:
                print(f"     - {missing}")
        
        if semantic_result['incorrect_configurations']:
            print("  Incorrect:")
            for incorrect in semantic_result['incorrect_configurations']:
                print(f"     - {incorrect}")
        print()
        
        # Final Decision
        overall_valid = (
            syntax_valid and
            structure_result['structure_valid'] and
            semantic_result['matches_requirements'] and
            semantic_result['correctness_score'] >= 7
        )
        
        print("="*70)
        print("🎯 FINAL VALIDATION RESULT")
        print("="*70)
        print(f"Overall Status: {'✅ APPROVED' if overall_valid else '❌ REJECTED'}")
        print(f"Recommendation: {semantic_result['recommendation'].upper()}")
        print("="*70 + "\n")
        
        return {
            'valid': overall_valid,
            'syntax_validation': {
                'passed': syntax_valid,
                'errors': all_errors,
                'warnings': all_warnings
            },
            'structure_validation': structure_result,
            'best_practices': best_practices,
            'semantic_validation': semantic_result,
            'recommendation': semantic_result['recommendation'],
            'overall_score': (
                (10 if syntax_valid else 0) +
                structure_result['score'] +
                best_practices['best_practices_score'] +
                semantic_result['correctness_score']
            ) / 4
        }


class RAGSystemWithChaining:
    def __init__(self, pinecone_api_key, index_name="terraform-iac-v1", 
                 generation_model="codellama:7b-instruct",
                 output_dir="terraform_output"):
        """
        Initialize RAG system with Pinecone and Ollama with prompt chaining
        """
        self.generation_model = generation_model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        
        print(f"✓ Connected to Pinecone index: {index_name}")
        print(f"✓ Using embedding model: all-MiniLM-L6-v2")
        print(f"✓ Using generation model: {generation_model}")
        print(f"✓ Output directory: {output_dir}\n")
    
    def get_embedding(self, text):
        """Generate embeddings using SentenceTransformer"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents from Pinecone"""
        query_embedding = self.get_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
    
    def format_context(self, documents):
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            text = metadata.get('text', 'No content available')
            score = doc.get('score', 0)
            context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{text}")
        return "\n\n".join(context_parts)
    
    def extract_requirements(self, user_query):
        """Step 1: Extract what needs to be created"""
        print("🔗 STEP 1: Analyzing your request...\n")
        
        prompt = f"""Analyze this Terraform infrastructure request and extract:
1. Resource type (e.g., S3 bucket, EC2 instance, VPC)
2. Required variables that MUST be collected from the user
3. Optional configurations mentioned

User Request: {user_query}

Respond in this exact JSON format:
{{
    "resource_type": "the main resource to create",
    "required_variables": ["list", "of", "required", "variables"],
    "optional_configs": ["list", "of", "optional", "settings"],
    "clarification_needed": true/false
}}
"""
        
        response = ollama.generate(model=self.generation_model, prompt=prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
            else:
                requirements = {
                    "resource_type": "infrastructure",
                    "required_variables": [],
                    "optional_configs": [],
                    "clarification_needed": True
                }
        except:
            requirements = {
                "resource_type": "infrastructure",
                "required_variables": [],
                "optional_configs": [],
                "clarification_needed": True
            }
        
        print(f"📋 Resource Type: {requirements['resource_type']}")
        print(f"📋 Required Variables: {', '.join(requirements['required_variables']) if requirements['required_variables'] else 'None identified'}")
        print(f"📋 Optional Configs: {', '.join(requirements['optional_configs']) if requirements['optional_configs'] else 'None'}\n")
        
        return requirements
    
    def collect_user_variables(self, requirements):
        """Step 2: Interactively collect required variables"""
        print("🔗 STEP 2: Collecting required information...\n")
        
        user_variables = {}
        
        if not requirements['required_variables']:
            print("✓ No additional variables needed\n")
            return user_variables
        
        print("Please provide the following information:\n")
        
        for var in requirements['required_variables']:
            value = input(f"  {var}: ").strip()
            user_variables[var] = value
        
        if requirements['optional_configs']:
            print("\n📝 Optional configurations (press Enter to skip):\n")
            for config in requirements['optional_configs']:
                value = input(f"  {config} [optional]: ").strip()
                if value:
                    user_variables[config] = value
        
        print(f"\n✓ Collected {len(user_variables)} variable(s)\n")
        return user_variables
    
    def enrich_query_with_context(self, original_query, requirements, user_variables):
        """Step 3: Create enriched query"""
        print("🔗 STEP 3: Enriching query with collected information...\n")
        
        enriched_parts = [
            f"Resource Type: {requirements['resource_type']}",
            f"Original Request: {original_query}"
        ]
        
        if user_variables:
            enriched_parts.append("\nUser-Provided Variables:")
            for key, value in user_variables.items():
                enriched_parts.append(f"  - {key}: {value}")
        
        enriched_query = "\n".join(enriched_parts)
        print("✓ Enriched query created\n")
        return enriched_query
    
    def generate_terraform_code(self, enriched_query, context, user_variables):
        """Step 4: Generate final Terraform code"""
        print("🔗 STEP 4: Generating Terraform code...\n")
        
        variables_section = ""
        if user_variables:
            variables_section = "\n\nUser Variables:\n"
            for key, value in user_variables.items():
                variables_section += f"  {key} = {value}\n"
        
        augmented_prompt = f"""You are a DevOps assistant expert in Terraform IaC.

CRITICAL REQUIREMENTS:
1. Generate a COMPLETE Terraform configuration
2. Return ONLY a valid JSON object in this exact format:
{{
    "main_tf": "complete main.tf content with terraform, provider, and resource blocks",
    "variables_tf": "variables.tf content with variable declarations",
    "outputs_tf": "outputs.tf content with output blocks",
    "terraform_tfvars": "terraform.tfvars content with variable values"
}}

3. Each JSON value should contain complete, valid Terraform HCL code
4. Use proper Terraform syntax and formatting
5. Include comments for clarity
6. Make the code production-ready
7. Do NOT add any text outside the JSON object

Retrieved Context from Knowledge Base:
{context}

Request Details:
{enriched_query}
{variables_section}

Generate the JSON response with all Terraform files:
"""
        
        response = ollama.generate(
            model=self.generation_model,
            prompt=augmented_prompt
        )
        
        return response['response']
    
    def clean_terraform_code(self, code):
        """Parse JSON response and extract Terraform files"""
        try:
            json_match = re.search(r'\{.*\}', code, re.DOTALL)
            if json_match:
                terraform_files = json.loads(json_match.group())
                
                default_files = {
                    "main_tf": "",
                    "variables_tf": "",
                    "outputs_tf": "",
                    "terraform_tfvars": ""
                }
                
                for key in default_files:
                    if key not in terraform_files:
                        terraform_files[key] = default_files[key]
                
                return terraform_files
            else:
                return {
                    "main_tf": code,
                    "variables_tf": "",
                    "outputs_tf": "",
                    "terraform_tfvars": ""
                }
        except json.JSONDecodeError:
            return {
                "main_tf": code,
                "variables_tf": "",
                "outputs_tf": "",
                "terraform_tfvars": ""
            }
    
    def save_terraform_file(self, code, resource_type, user_variables):
        """Save generated Terraform code to multiple files"""
        terraform_files = self.clean_terraform_code(code)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_resource = re.sub(r'[^\w\-]', '_', resource_type.lower())
        base_name = f"{sanitized_resource}_{timestamp}"
        
        saved_files = {}
        
        if terraform_files.get('main_tf') and terraform_files['main_tf'].strip():
            main_filepath = os.path.join(self.output_dir, f"{base_name}_main.tf")
            with open(main_filepath, 'w') as f:
                f.write(terraform_files['main_tf'])
            saved_files['main_tf'] = main_filepath
        
        if terraform_files.get('variables_tf') and terraform_files['variables_tf'].strip():
            vars_filepath = os.path.join(self.output_dir, f"{base_name}_variables.tf")
            with open(vars_filepath, 'w') as f:
                f.write(terraform_files['variables_tf'])
            saved_files['variables_tf'] = vars_filepath
        
        if terraform_files.get('outputs_tf') and terraform_files['outputs_tf'].strip():
            outputs_filepath = os.path.join(self.output_dir, f"{base_name}_outputs.tf")
            with open(outputs_filepath, 'w') as f:
                f.write(terraform_files['outputs_tf'])
            saved_files['outputs_tf'] = outputs_filepath
        
        if terraform_files.get('terraform_tfvars') and terraform_files['terraform_tfvars'].strip():
            tfvars_filepath = os.path.join(self.output_dir, f"{base_name}.tfvars")
            with open(tfvars_filepath, 'w') as f:
                f.write(terraform_files['terraform_tfvars'])
            saved_files['terraform_tfvars'] = tfvars_filepath
        elif user_variables:
            tfvars_filepath = os.path.join(self.output_dir, f"{base_name}.tfvars")
            with open(tfvars_filepath, 'w') as f:
                f.write("# Auto-generated Terraform variables file\n\n")
                for key, value in user_variables.items():
                    if not value.startswith('"'):
                        value = f'"{value}"'
                    f.write(f'{key} = {value}\n')
            saved_files['terraform_tfvars'] = tfvars_filepath
        
        json_filepath = os.path.join(self.output_dir, f"{base_name}_complete.json")
        with open(json_filepath, 'w') as f:
            json.dump(terraform_files, f, indent=2)
        saved_files['json'] = json_filepath
        
        return saved_files


class RAGSystemWithValidation(RAGSystemWithChaining):
    """
    Extended RAG system with validation loop
    """
    def __init__(self, pinecone_api_key, index_name="terraform-iac-v1", 
                 generation_model="codellama:7b-instruct",
                 output_dir="terraform_output",
                 max_regeneration_attempts=3):
        super().__init__(pinecone_api_key, index_name, generation_model, output_dir)
        self.validator = TerraformValidationAgent(generation_model)
        self.max_regeneration_attempts = max_regeneration_attempts
    
    def generate_terraform_with_feedback(self, enriched_query, context, user_variables, previous_issues=None):
        """
        Generate Terraform code with feedback from previous validation
        """
        feedback_section = ""
        if previous_issues:
            feedback_section = f"""

IMPORTANT - PREVIOUS ATTEMPT HAD ISSUES:
Syntax Errors: {', '.join(previous_issues.get('syntax_errors', [])) if previous_issues.get('syntax_errors') else 'None'}
Structure Issues: {', '.join(previous_issues.get('structure_issues', [])) if previous_issues.get('structure_issues') else 'None'}
Semantic Issues: {', '.join(previous_issues.get('semantic_issues', [])) if previous_issues.get('semantic_issues') else 'None'}
Missing Components: {', '.join(previous_issues.get('missing_components', [])) if previous_issues.get('missing_components') else 'None'}

YOU MUST FIX ALL OF THE ABOVE ISSUES IN THIS GENERATION.
"""
        
        variables_section = ""
        if user_variables:
            variables_section = "\n\nUser Variables:\n"
            for key, value in user_variables.items():
                variables_section += f"  {key} = {value}\n"
        
        augmented_prompt = f"""You are a DevOps assistant expert in Terraform IaC.

CRITICAL REQUIREMENTS:
1. Generate a COMPLETE, VALID Terraform configuration
2. Return ONLY a valid JSON object in this exact format:
{{
    "main_tf": "complete main.tf content with terraform, provider, and resource blocks",
    "variables_tf": "variables.tf content with variable declarations",
    "outputs_tf": "outputs.tf content with output blocks",
    "terraform_tfvars": "terraform.tfvars content with variable values"
}}

3. Each JSON value must contain complete, valid Terraform HCL code
4. Use proper Terraform syntax (balanced braces, quotes, brackets)
5. Include ALL required blocks: terraform, required_providers, provider, and resource
6. Add proper version constraints for providers
7. Include comprehensive comments
8. Make the code production-ready
9. Do NOT add any text outside the JSON object
{feedback_section}

Retrieved Context from Knowledge Base:
{context}

Request Details:
{enriched_query}
{variables_section}

Generate the JSON response with all Terraform files now:
"""
        
        response = ollama.generate(
            model=self.generation_model,
            prompt=augmented_prompt
        )
        
        return response['response']
    
    def query_with_validation_loop(self, user_query, top_k=3, auto_collect=True, save_to_file=True):
        """
        Complete RAG pipeline with validation and regeneration loop
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting RAG Pipeline with Validation Loop")
        print(f"{'='*70}\n")
        print(f"Initial Query: {user_query}\n")
        
        # STEP 1: Extract requirements
        requirements = self.extract_requirements(user_query)
        
        # STEP 2: Collect variables
        if auto_collect:
            user_variables = self.collect_user_variables(requirements)
        else:
            return {
                'requirements': requirements,
                'status': 'awaiting_user_input'
            }
        
        # STEP 3: Enrich query
        enriched_query = self.enrich_query_with_context(
            user_query, requirements, user_variables
        )
        
        # STEP 4: Retrieve documents
        print("🔗 STEP 5: Retrieving relevant documentation...\n")
        documents = self.retrieve_documents(enriched_query, top_k=top_k)
        context = self.format_context(documents)
        print(f"✓ Retrieved {len(documents)} relevant documents\n")
        
        # VALIDATION LOOP
        attempt = 0
        terraform_code = None
        validation_result = None
        previous_issues = None
        
        while attempt < self.max_regeneration_attempts:
            attempt += 1
            print(f"\n{'='*70}")
            print(f"🔄 GENERATION ATTEMPT {attempt}/{self.max_regeneration_attempts}")
            print(f"{'='*70}\n")
            
            # Generate code
            if attempt == 1:
                terraform_code = self.generate_terraform_code(
                    enriched_query, context, user_variables
                )
            else:
                print("♻️  Regenerating with feedback from validation...\n")
                terraform_code = self.generate_terraform_with_feedback(
                    enriched_query, context, user_variables, previous_issues
                )
            
            print(f"{'='*70}")
            print("📄 GENERATED TERRAFORM CODE")
            print(f"{'='*70}\n")
            print(terraform_code[:500] + "..." if len(terraform_code) > 500 else terraform_code)
            print(f"\n{'='*70}")
            
            # Validate generated code
            validation_result = self.validator.comprehensive_validation(
                terraform_code,
                user_query,
                user_variables
            )
            
            # Check if validation passed
            if validation_result['valid']:
                print("✅ Validation PASSED! Code is ready to use.\n")
                break
            else:
                print(f"❌ Validation FAILED on attempt {attempt}")
                
                if attempt < self.max_regeneration_attempts:
                    print(f"🔄 Will retry with corrections...\n")
                    
                    # Prepare feedback for next iteration
                    previous_issues = {
                        'syntax_errors': validation_result['syntax_validation']['errors'],
                        'structure_issues': validation_result['structure_validation']['issues'],
                        'semantic_issues': validation_result['semantic_validation']['issues_found'],
                        'missing_components': validation_result['semantic_validation']['missing_components']
                    }
                else:
                    print(f"⚠️  Maximum attempts ({self.max_regeneration_attempts}) reached.\n")
        
        # Save to file if validation passed or max attempts reached
        saved_files = None
        if save_to_file and validation_result:
            if validation_result['valid'] or attempt >= self.max_regeneration_attempts:
                print("💾 Saving generated files...")
                saved_files = self.save_terraform_file(
                    terraform_code, 
                    requirements['resource_type'],
                    user_variables
                )
                
                for file_type, filepath in saved_files.items():
                    print(f"  ✓ {file_type}: {filepath}")
                print()
        
        return {
            'terraform_code': terraform_code,
            'requirements': requirements,
            'user_variables': user_variables,
            'documents': documents,
            'validation_result': validation_result,
            'saved_files': saved_files,
            'attempts': attempt,
            'status': 'completed' if validation_result['valid'] else 'failed_validation',
            'final_score': validation_result['overall_score']
        }
    
    def batch_query_with_variables(self, user_query, variables_dict, top_k=3, save_to_file=True):
        """
        Non-interactive mode with validation loop
        """
        print(f"\n{'='*70}")
        print(f"🚀 Batch Mode RAG Pipeline with Validation")
        print(f"{'='*70}\n")
        
        requirements = self.extract_requirements(user_query)
        enriched_query = self.enrich_query_with_context(
            user_query, requirements, variables_dict
        )
        
        print("🔗 Retrieving relevant documentation...\n")
        documents = self.retrieve_documents(enriched_query, top_k=top_k)
        context = self.format_context(documents)
        print(f"✓ Retrieved {len(documents)} relevant documents\n")
        
        # VALIDATION LOOP for batch mode
        attempt = 0
        terraform_code = None
        validation_result = None
        previous_issues = None
        
        while attempt < self.max_regeneration_attempts:
            attempt += 1
            print(f"\n{'='*70}")
            print(f"🔄 GENERATION ATTEMPT {attempt}/{self.max_regeneration_attempts}")
            print(f"{'='*70}\n")
            
            if attempt == 1:
                terraform_code = self.generate_terraform_code(
                    enriched_query, context, variables_dict
                )
            else:
                print("♻️  Regenerating with feedback...\n")
                terraform_code = self.generate_terraform_with_feedback(
                    enriched_query, context, variables_dict, previous_issues
                )
            
            validation_result = self.validator.comprehensive_validation(
                terraform_code,
                user_query,
                variables_dict
            )
            
            if validation_result['valid']:
                print("✅ Validation PASSED!\n")
                break
            else:
                print(f"❌ Validation FAILED on attempt {attempt}")
                if attempt < self.max_regeneration_attempts:
                    previous_issues = {
                        'syntax_errors': validation_result['syntax_validation']['errors'],
                        'structure_issues': validation_result['structure_validation']['issues'],
                        'semantic_issues': validation_result['semantic_validation']['issues_found'],
                        'missing_components': validation_result['semantic_validation']['missing_components']
                    }
        
        saved_files = None
        if save_to_file and validation_result:
            print("💾 Saving to files...")
            saved_files = self.save_terraform_file(
                terraform_code,
                requirements['resource_type'],
                variables_dict
            )
            
            for file_type, filepath in saved_files.items():
                print(f"  ✓ {file_type}: {filepath}")
            print()
        
        return {
            'terraform_code': terraform_code,
            'requirements': requirements,
            'user_variables': variables_dict,
            'documents': documents,
            'validation_result': validation_result,
            'saved_files': saved_files,
            'attempts': attempt,
            'status': 'completed' if validation_result['valid'] else 'failed_validation',
            'final_score': validation_result['overall_score']
        }


# Example usage
if __name__ == "__main__":
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key-here")
    
    # Initialize RAG system with validation
    rag = RAGSystemWithValidation(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="terraform-iac-v1",
        output_dir="generated_terraform",
        max_regeneration_attempts=3
    )
    
    # Example 1: Interactive mode with validation loop
    print("="*70)
    print("EXAMPLE 1: Interactive Mode with Validation")
    print("="*70)
    
    result = rag.query_with_validation_loop(
        "Create an S3 bucket with versioning, encryption, and lifecycle rules",
        save_to_file=True
    )
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Final Score: {result['final_score']:.2f}/10")
    print(f"Valid: {result['validation_result']['valid']}")
    
    if result['saved_files']:
        print("\n💾 Saved Files:")
        for file_type, path in result['saved_files'].items():
            print(f"  - {file_type}: {path}")
    
    # Example 2: Batch mode with pre-defined variables
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Batch Mode with Validation")
    print("="*70)
    
    batch_result = rag.batch_query_with_variables(
        user_query="Create an EC2 instance",
        variables_dict={
            "instance_type": "t3.micro",
            "ami_id": "ami-0c55b159cbfafe1f0",
            "region": "us-east-1"
        },
        save_to_file=True
    )
    
    print("\n" + "="*70)
    print("📊 BATCH RESULTS")
    print("="*70)
    print(f"Status: {batch_result['status']}")
    print(f"Attempts: {batch_result['attempts']}")
    print(f"Final Score: {batch_result['final_score']:.2f}/10")
    print(f"Valid: {batch_result['validation_result']['valid']}")