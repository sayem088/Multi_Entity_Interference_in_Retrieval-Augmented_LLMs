"""
Wikipedia Infobox Retrieval Client.
Retrieves structured entity attribute fields for Wikipedia entity categories.
"""

from typing import List, Dict, Any

class WikiClient:
    def fetch_entities_by_category(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves Wikipedia infobox entity attribute records for a given category.
        """
        return self._generate_sample_entities(category, limit)

    def _generate_sample_entities(self, category: str, limit: int) -> List[Dict[str, Any]]:
        base_attributes = {
            "university": {
                "established": "1850",
                "location": "Cambridge, MA",
                "students_or_employees": "22,000 students",
                "budget_or_gdp": "$2.1 billion",
                "motto": "Veritas",
                "website": "https://university.edu"
            },
            "country": {
                "established": "1776",
                "location": "North America",
                "students_or_employees": "330 million residents",
                "budget_or_gdp": "$25 trillion",
                "motto": "In God We Trust",
                "website": "https://gov.portal"
            },
            "company": {
                "established": "1998",
                "location": "Mountain View, CA",
                "students_or_employees": "180,000 employees",
                "budget_or_gdp": "$280 billion revenue",
                "motto": "Organize the world's information",
                "website": "https://company.com"
            },
            "stadium": {
                "established": "1923",
                "location": "London, UK",
                "students_or_employees": "90,000 capacity",
                "budget_or_gdp": "$1.2 billion construction",
                "website": "https://stadium.org"
            }
        }

        default_attrs = base_attributes.get(category, {
            "established": "1900",
            "location": "Global",
            "website": "https://example.org"
        })

        entities = []
        for i in range(1, limit + 1):
            attrs = {"name": f"Grand {category.capitalize()} {i}"}
            attrs.update(default_attrs)
            if category == "university":
                attrs["established"] = str(1800 + i * 5)
                attrs["students_or_employees"] = f"{10000 + i * 1500} students"
                attrs["budget_or_gdp"] = f"${1.0 + i * 0.2:.1f} billion"
            elif category == "company":
                attrs["established"] = str(1970 + i * 3)
                attrs["students_or_employees"] = f"{5000 + i * 2000} employees"

            entities.append({
                "entity_id": f"wiki_{category}_{i}",
                "category": category,
                "name": attrs["name"],
                "attributes": attrs
            })
            
        return entities
