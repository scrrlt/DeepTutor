import pytest
from unittest.mock import patch, MagicMock
from src.agents.ideagen.material_organizer_agent import MaterialOrganizerAgent

class TestMaterialOrganizerAgent:
    
    @pytest.fixture
    def agent(self):
        return MaterialOrganizerAgent(
            llm_client=MagicMock(),
            config={"organization_style": "hierarchical"}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert agent.config["organization_style"] == "hierarchical"
        
    @patch('src.agents.ideagen.material_organizer_agent.MaterialOrganizerAgent._organize_materials')
    def test_organize_materials(self, mock_organize, agent):
        materials = ["Material 1", "Material 2", "Material 3"]
        mock_organize.return_value = {"organized_structure": "Structured materials"}
        
        result = agent.organize_materials(materials)
        
        assert result["organized_structure"] == "Structured materials"
        mock_organize.assert_called_once_with(materials)
        
    @patch('src.agents.ideagen.material_organizer_agent.MaterialOrganizerAgent._categorize_material')
    def test_categorize_material(self, mock_categorize, agent):
        material = "Material 1"
        mock_categorize.return_value = "Category 1"

        result = agent._categorize_material(material)

        assert result == "Category 1"
        mock_categorize.assert_called_once_with(material)
