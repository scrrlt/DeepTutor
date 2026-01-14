"""
Default Configuration Values
===========================

Default configuration values for DeepTutor.
These are merged with user configuration from main.yaml.
"""

from typing import Any, Dict

# Default configuration values
DEFAULTS: Dict[str, Any] = {
    "system": {
        "language": "en",
    },
    "paths": {
        "user_data_dir": "./data/user",
        "knowledge_bases_dir": "./data/knowledge_bases",
        "user_log_dir": "./data/user/logs",
        "performance_log_dir": "./data/user/performance",
        "guide_output_dir": "./data/user/guide",
        "question_output_dir": "./data/user/question",
        "research_output_dir": "./data/user/research/cache",
        "research_reports_dir": "./data/user/research/reports",
        "solve_output_dir": "./data/user/solve",
    },
    "llm": {
        "model": "gpt-4.1-mini",  # Default model
        "provider": "openai",
    },
    "tools": {
        "rag_tool": {
            "kb_base_dir": "./data/knowledge_bases",
            "default_kb": "ai_textbook",
        },
        "run_code": {
            "workspace": "./data/user/run_code_workspace",
            "allowed_roots": [
                "./data/user",
                "./src/tools",
            ],
        },
        "web_search": {
            "enabled": True,
        },
        "query_item": {
            "enabled": True,
            "max_results": 5,
        },
    },
    "logging": {
        "level": "INFO",
        "save_to_file": True,
        "console_output": True,
        "lightrag_forwarding": {
            "enabled": True,
            "min_level": "DEBUG",
            "add_prefix": True,
            "logger_names": {
                "knowledge_init": "LightRAG-Init",
                "rag_tool": "LightRAG",
            },
        },
    },
    "tts": {
        "default_voice": "alloy",
    },
    "question": {
        "max_rounds": 10,
        "rag_query_count": 3,
        "max_parallel_questions": 1,
        "rag_mode": "naive",
        "agents": {
            "question_generation": {
                "max_iterations": 5,
                "retrieve_top_k": 30,
            },
            "question_validation": {
                "strict_mode": True,
            },
        },
    },
    "solve": {
        "max_solve_correction_iterations": 3,
        "enable_citations": True,
        "save_intermediate_results": True,
        "valid_tools": [
            "web_search",
            "code_execution",
            "rag_naive",
            "rag_hybrid",
            "query_item",
            "none",
            "finish",
        ],
        "agents": {
            "investigate_agent": {
                "max_actions_per_round": 1,
                "max_iterations": 3,
            },
            "precision_answer_agent": {
                "enabled": True,
            },
        },
    },
    "research": {
        "planning": {
            "rephrase": {
                "enabled": True,
                "max_iterations": 3,
            },
            "decompose": {
                "enabled": True,
                "mode": "auto",
                "initial_subtopics": 5,
                "auto_max_subtopics": 8,
            },
        },
        "researching": {
            "max_iterations": 5,
            "new_topic_min_score": 0.85,
            "execution_mode": "series",
            "max_parallel_topics": 1,
            "iteration_mode": "fixed",
            "enable_rag_naive": True,
            "enable_rag_hybrid": True,
            "enable_paper_search": True,
            "enable_web_search": True,
            "enable_run_code": True,
            "tool_timeout": 60,
            "tool_max_retries": 2,
            "paper_search_years_limit": 3,
        },
        "reporting": {
            "min_section_length": 800,
            "enable_citation_list": True,
            "enable_inline_citations": False,
        },
        "rag": {
            "kb_name": "DE-all",
            "default_mode": "hybrid",
            "fallback_mode": "naive",
        },
        "queue": {
            "max_length": 5,
        },
        "presets": {
            "quick": {
                "description": "Quick mode - fast research with minimal depth",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 1,
                    },
                },
                "researching": {
                    "max_iterations": 1,
                    "iteration_mode": "fixed",
                },
                "reporting": {
                    "min_section_length": 300,
                },
            },
            "medium": {
                "description": "Medium mode - balanced research depth",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 5,
                    },
                },
                "researching": {
                    "max_iterations": 4,
                    "iteration_mode": "fixed",
                },
                "reporting": {
                    "min_section_length": 500,
                },
            },
            "deep": {
                "description": "Deep mode - thorough research with maximum depth",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 8,
                    },
                },
                "researching": {
                    "max_iterations": 7,
                    "iteration_mode": "fixed",
                },
                "reporting": {
                    "min_section_length": 800,
                },
            },
            "auto": {
                "description": "Auto mode - agent decides depth within limits",
                "planning": {
                    "decompose": {
                        "mode": "auto",
                        "auto_max_subtopics": 8,
                    },
                },
                "researching": {
                    "max_iterations": 6,
                    "iteration_mode": "flexible",
                },
                "reporting": {
                    "min_section_length": 500,
                },
            },
        },
    },
}
