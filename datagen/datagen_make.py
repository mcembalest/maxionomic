from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable
import random
import datetime
import uuid
from enum import Enum
import json
from abc import ABC, abstractmethod
from pathlib import Path

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    NESTED = "nested"

@dataclass
class FieldDefinition:
    field_type: FieldType
    required: bool = True
    options: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    list_size_range: Optional[tuple[int, int]] = None
    custom_generator: Optional[Callable] = None
    nested_schema: Optional[Dict] = None

@dataclass
class SchemaDefinition:
    fields: Dict[str, FieldDefinition]
    probability_present: float = 1.0

class Schema:
    """Simplified schema builder with better syntax"""
    def __init__(self, fields: Dict[str, dict]):
        self.fields = {
            name: FieldDefinition(**specs) if isinstance(specs, dict) else specs
            for name, specs in fields.items()
        }

    @staticmethod
    def string() -> Dict:
        return {"field_type": FieldType.STRING}
    
    @staticmethod
    def integer(min_val: Optional[int] = None, max_val: Optional[int] = None) -> Dict:
        return {"field_type": FieldType.INTEGER, "min_value": min_val, "max_value": max_val}
    
    @staticmethod
    def float(min_val: Optional[float] = None, max_val: Optional[float] = None) -> Dict:
        return {"field_type": FieldType.FLOAT, "min_value": min_val, "max_value": max_val}
    
    @staticmethod
    def date() -> Dict:
        return {"field_type": FieldType.DATE}
    
    @staticmethod
    def enum(options: List[str]) -> Dict:
        return {"field_type": FieldType.ENUM, "options": options}
    
    @staticmethod
    def list_of(items: Dict, min_size: int = 0, max_size: int = 5) -> Dict:
        return {
            "field_type": FieldType.LIST,
            "list_size_range": (min_size, max_size),
            "nested_schema": items
        }
    
    @staticmethod
    def nested(schema: Dict) -> Dict:
        return {"field_type": FieldType.NESTED, "nested_schema": schema}


class Generator:
    """Simplified generator with fixed nested structure handling"""
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
    
    def generate(self, schema: Schema, num_samples: int = 1000) -> List[Dict]:
        return [self._generate_object(schema.fields) for _ in range(num_samples)]
    
    def _generate_object(self, fields: Dict[str, FieldDefinition]) -> Dict:
        result = {}
        for name, field in fields.items():
            value = self._generate_value(field)
            if value is not None:  # Only add non-null values
                result[name] = value
        return result
    
    def _generate_value(self, field: FieldDefinition) -> any:
        # Only check required at top level, not for nested structures
        if not field.required and not (field.field_type in [FieldType.LIST, FieldType.NESTED]) and self.random.random() > 0.8:
            return None
            
        if field.custom_generator:
            return field.custom_generator(self.random)
            
        try:
            if field.field_type == FieldType.STRING:
                return str(uuid.uuid4())
            elif field.field_type == FieldType.INTEGER:
                return self.random.randint(field.min_value or 0, field.max_value or 100)
            elif field.field_type == FieldType.FLOAT:
                return round(self.random.uniform(field.min_value or 0.0, field.max_value or 1.0), 2)
            elif field.field_type == FieldType.DATE:
                days = self.random.randint(0, 365 * 5)
                return (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            elif field.field_type == FieldType.BOOLEAN:
                return self.random.choice([True, False])
            elif field.field_type == FieldType.ENUM and field.options:
                return self.random.choice(field.options)
            elif field.field_type == FieldType.LIST:
                size = self.random.randint(*field.list_size_range or (0, 5))
                items = [self._generate_value(FieldDefinition(**field.nested_schema)) for _ in range(size)]
                return [item for item in items if item is not None]  # Filter out any null items
            elif field.field_type == FieldType.NESTED:
                nested_obj = self._generate_object(SchemaDefinition(field.nested_schema).fields)
                return nested_obj if nested_obj else None  # Return None if nested object is empty
        except Exception as e:
            print(f"Error generating value for field type {field.field_type}: {e}")
            return None

# Data definitions moved to separate classes for clarity
class FinancialData:
    COMPANIES = [
        "Acme Global", "TechCorp Solutions", "Quantum Industries",
        "Atlas Technologies", "Pinnacle Systems", "Nova Enterprises",
        "Summit Corp", "Vertex Holdings", "Axiom Innovations"
    ]
    TYPES = ["Public", "Private", "Subsidiary"]
    DIVISIONS = [
        "Research & Development", "Sales & Marketing", "Operations",
        "Human Resources", "Finance", "Information Technology",
        "Product Development", "Customer Service", "Manufacturing"
    ]

class LegalData:
    JURISDICTIONS = [
        "New York Southern", "California Northern", "Texas Eastern",
        "Florida Middle", "Illinois Northern", "Massachusetts"
    ]
    TYPES = ["Federal", "State", "Administrative"]
    CASE_TYPES = [
        "Civil Rights", "Contract Dispute", "Employment",
        "Intellectual Property", "Securities", "Antitrust"
    ]
    FILING_TYPES = [
        "Complaint", "Motion to Dismiss", "Answer",
        "Discovery Request", "Summary Judgment", "Appeal"
    ]

class ProductData:
    PRODUCTS = [
        "Smart Watch Pro", "Wireless Earbuds", "Laptop Elite",
        "Gaming Mouse", "4K Monitor", "Mechanical Keyboard"
    ]
    CATEGORIES = [
        "Electronics", "Computing", "Audio",
        "Gaming", "Photography", "Accessories"
    ]
    COMMENTS = [
        "Great product!", "Excellent value", "Highly recommended",
        "Good quality", "Could be better", "Amazing features"
    ]

class Generator:
    """Generator with proper nested structure handling"""
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
    
    def generate(self, schema: Schema, num_samples: int = 1000) -> List[Dict]:
        return [self._generate_object(schema.fields) for _ in range(num_samples)]
    
    def _generate_object(self, fields: Dict[str, FieldDefinition]) -> Dict:
        result = {}
        for name, field in fields.items():
            value = self._generate_value(field)
            if value is not None:
                result[name] = value
        return result
    
    def _generate_value(self, field: FieldDefinition) -> any:
        try:
            if field.field_type == FieldType.STRING:
                return str(uuid.uuid4())
            elif field.field_type == FieldType.INTEGER:
                return self.random.randint(field.min_value or 0, field.max_value or 100)
            elif field.field_type == FieldType.FLOAT:
                return round(self.random.uniform(field.min_value or 0.0, field.max_value or 1.0), 2)
            elif field.field_type == FieldType.DATE:
                days = self.random.randint(0, 365 * 5)
                return (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            elif field.field_type == FieldType.BOOLEAN:
                return self.random.choice([True, False])
            elif field.field_type == FieldType.ENUM and field.options:
                return self.random.choice(field.options)
            elif field.field_type == FieldType.LIST:
                size = self.random.randint(*field.list_size_range or (0, 5))
                items = []
                for _ in range(size):
                    # Convert nested schema to FieldDefinition
                    nested_field = FieldDefinition(**field.nested_schema)
                    item = self._generate_value(nested_field)
                    if item is not None:
                        items.append(item)
                return items
            elif field.field_type == FieldType.NESTED:
                # Convert nested schema to proper field definitions
                nested_fields = {
                    k: FieldDefinition(**v) if isinstance(v, dict) else v
                    for k, v in field.nested_schema.items()
                }
                return self._generate_object(nested_fields)
        except Exception as e:
            print(f"Error generating value for field type {field.field_type}")
            raise e

def generate_datasets(output_dir: str = "datagen_output", num_samples: int = 1000):
    generator = Generator(seed=42)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Financial Data Schema
    financial_schema = Schema({
        "company_id": Schema.string(),
        "name": Schema.enum(FinancialData.COMPANIES),
        "type": Schema.enum(FinancialData.TYPES),
        "divisions": Schema.list_of(Schema.nested({
            "name": Schema.enum(FinancialData.DIVISIONS),
            "budget": Schema.float(1000000, 10000000),
            "projects": Schema.list_of(Schema.nested({
                "id": Schema.string(),
                "amount": Schema.float(100000, 1000000)
            }))
        }))
    })

    # Legal Data Schema
    legal_schema = Schema({
        "system_id": Schema.string(),
        "jurisdiction": Schema.enum(LegalData.JURISDICTIONS),
        "type": Schema.enum(LegalData.TYPES),
        "cases": Schema.list_of(Schema.nested({
            "number": Schema.string(),
            "type": Schema.enum(LegalData.CASE_TYPES),
            "filings": Schema.list_of(Schema.nested({
                "id": Schema.string(),
                "type": Schema.enum(LegalData.FILING_TYPES),
                "date": Schema.date()
            }))
        }))
    })

    # Product Data Schema
    product_schema = Schema({
        "id": Schema.string(),
        "name": Schema.enum(ProductData.PRODUCTS),
        "price": Schema.float(99.99, 1999.99),
        "categories": Schema.list_of(Schema.enum(ProductData.CATEGORIES)),
        "reviews": Schema.list_of(Schema.nested({
            "user_id": Schema.string(),
            "rating": Schema.integer(1, 5),
            "comment": Schema.enum(ProductData.COMMENTS),
            "date": Schema.date()
        }))
    })

    schemas = {
        "financial": financial_schema,
        "legal": legal_schema,
        "product": product_schema
    }

    for name, schema in schemas.items():
        try:
            print(f"Generating {name} data...")
            data = generator.generate(schema, num_samples)
            output_file = output_path / f"{name}_data.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Generated {len(data)} {name} records")
            print(f"\nSample {name} record:")
            print(json.dumps(data[0], indent=2))
        except Exception as e:
            print(f"Error generating {name} data: {e}")

if __name__ == "__main__":
    generate_datasets()