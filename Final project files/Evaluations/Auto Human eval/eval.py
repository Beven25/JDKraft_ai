from config import *
from langchain.llms import CTransformers
from langchain.evaluation import load_evaluator

class MistralEvaluator:
    
    def __init__(self):
        
        self.model = MODEL
        self.model_type = MODEL_TYPE
        self.max_new_tokens = 2048
        self.temperature = TEMPERATURE
        self.do_sample = DO_SAMPLE
        self.context_length = 4096
        
        self.llm = CTransformers(
            model = self.model,
            model_type = self.model_type,
            max_new_tokens = self.max_new_tokens,
            temperature = self.temperature,
            do_sample = self.do_sample,
            config={'max_new_tokens': self.max_new_tokens,
                        'context_length': self.context_length}
                        )

        
        
        self.criterias = [CONCISENESS,RELEVANCE]
    
    def evaluate_criteria(self, criteria, input, prediction):
        
        evaluator = load_evaluator(
                                    evaluator="criteria", 
                                    criteria=criteria,
                                    llm=self.llm
                                )
        
        eval_result = evaluator.evaluate_strings(
                                                    input=input,
                                                    prediction=prediction,
                                                    # reference=reference
                                                )
        return eval_result["reasoning"]
    
    def eval_sample(self, input, prediction):
        
        eval_output = {}
        
        for criteria in self.criterias:
            criteria_eval_sample = self.evaluate_criteria(
                                                            criteria=criteria,
                                                            input=input, 
                                                            prediction=prediction,
                                                            # reference=reference
                                                        )
            eval_output[criteria] = criteria_eval_sample
        
        return eval_output