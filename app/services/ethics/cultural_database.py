"""
Cultural Database for Cultural Sensitivity Service

This module provides a comprehensive database of cultural garments,
their significance, appropriate contexts, and cultural information.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CulturalItem:
    """Cultural item data structure"""
    name: str
    culture: str
    region: str
    significance: str
    sacred_level: str
    appropriate_contexts: List[str]
    inappropriate_contexts: List[str]
    description: str
    typical_colors: List[str]
    traditional_patterns: List[str]
    historical_context: str
    modern_adaptations_acceptable: bool
    expert_consultation_required: bool
    regional_variations: List[str]


class CulturalDatabase:
    """
    Comprehensive cultural database for garments and accessories
    
    Maintains database of cultural items with their significance,
    appropriate usage contexts, and cultural sensitivity information.
    """
    
    def __init__(self):
        """Initialize cultural database with comprehensive data"""
        self.logger = logging.getLogger(__name__)
        
        # Cultural items database
        self.cultural_items = {}
        
        # Cultural education content
        self.educational_content = {}
        
        # Regional variations
        self.regional_variations = {}
        
        # Initialize with comprehensive cultural data
        self._initialize_cultural_database()
        
        self.logger.info(f"Cultural Database initialized with {len(self.cultural_items)} items")
    
    def _initialize_cultural_database(self):
        """Initialize database with comprehensive cultural items"""
        try:
            # Initialize with core cultural items
            self._load_asian_cultural_items()
            self._load_african_cultural_items()
            self._load_indigenous_cultural_items()
            self._load_middle_eastern_cultural_items()
            self._load_european_cultural_items()
            self._load_latin_american_cultural_items()
            
            # Load educational content
            self._load_educational_content()
            
            self.logger.info("Cultural database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Cultural database initialization failed: {str(e)}")
            self._load_minimal_database()
    
    def _load_asian_cultural_items(self):
        """Load Asian cultural items"""
        asian_items = [
            CulturalItem(
                name="kimono",
                culture="japanese",
                region="japan",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["tea_ceremony", "cultural_celebration", "formal_event", "artistic_performance"],
                inappropriate_contexts=["casual_wear", "halloween_costume", "beach_wear"],
                description="Traditional Japanese garment with deep cultural significance",
                typical_colors=["red", "black", "white", "gold", "navy"],
                traditional_patterns=["cherry_blossom", "crane", "geometric", "floral"],
                historical_context="Worn by Japanese nobility and later adopted across social classes",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["furisode", "yukata", "tomesode", "houmongi"]
            ),
            CulturalItem(
                name="sari",
                culture="indian",
                region="south_asia",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["wedding", "religious_ceremony", "cultural_celebration", "formal_event"],
                inappropriate_contexts=["costume_party", "casual_daily_wear_by_non_indians"],
                description="Traditional Indian draped garment with regional variations",
                typical_colors=["red", "gold", "green", "blue", "purple", "orange"],
                traditional_patterns=["paisley", "floral", "geometric", "border_designs"],
                historical_context="Ancient garment mentioned in Vedic literature",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["banarasi", "kanjeevaram", "chiffon", "cotton"]
            ),
            CulturalItem(
                name="cheongsam_qipao",
                culture="chinese",
                region="china",
                significance="medium",
                sacred_level="low",
                appropriate_contexts=["formal_event", "cultural_celebration", "fashion"],
                inappropriate_contexts=["stereotypical_representation"],
                description="Traditional Chinese dress with form-fitting silhouette",
                typical_colors=["red", "gold", "black", "blue"],
                traditional_patterns=["dragon", "phoenix", "floral", "geometric"],
                historical_context="Evolved from Manchu origins to modern Chinese fashion",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["shanghai_style", "hong_kong_style", "modern_fusion"]
            ),
            CulturalItem(
                name="hanbok",
                culture="korean",
                region="korea",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["cultural_ceremony", "traditional_celebration", "formal_event"],
                inappropriate_contexts=["casual_fashion", "costume_party"],
                description="Traditional Korean dress with distinctive silhouette",
                typical_colors=["white", "pink", "blue", "yellow", "red"],
                traditional_patterns=["floral", "geometric", "nature_motifs"],
                historical_context="Worn by Korean royalty and commoners with class distinctions",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["court_hanbok", "commoner_hanbok", "modern_hanbok"]
            ),
            CulturalItem(
                name="ao_dai",
                culture="vietnamese",
                region="vietnam",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["cultural_celebration", "formal_event", "wedding"],
                inappropriate_contexts=["casual_wear", "inappropriate_styling"],
                description="Traditional Vietnamese long dress with cultural significance",
                typical_colors=["white", "red", "blue", "yellow"],
                traditional_patterns=["floral", "nature", "geometric"],
                historical_context="National dress of Vietnam with historical evolution",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["northern_style", "southern_style", "modern_ao_dai"]
            )
        ]
        
        for item in asian_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_african_cultural_items(self):
        """Load African cultural items"""
        african_items = [
            CulturalItem(
                name="kente_cloth",
                culture="akan_ghanaian",
                region="west_africa",
                significance="extremely_high",
                sacred_level="high",
                appropriate_contexts=["cultural_ceremony", "graduation", "cultural_celebration"],
                inappropriate_contexts=["casual_fashion", "costume", "inappropriate_commercial_use"],
                description="Sacred cloth with symbolic patterns and royal significance",
                typical_colors=["gold", "green", "red", "black"],
                traditional_patterns=["geometric", "symbolic", "royal_patterns"],
                historical_context="Woven by Akan people with spiritual and social significance",
                modern_adaptations_acceptable=False,
                expert_consultation_required=True,
                regional_variations=["asante_kente", "ewe_kente"]
            ),
            CulturalItem(
                name="dashiki",
                culture="west_african",
                region="west_africa",
                significance="medium",
                sacred_level="low",
                appropriate_contexts=["cultural_celebration", "casual_wear", "artistic_expression"],
                inappropriate_contexts=["stereotypical_representation", "costume"],
                description="Traditional West African garment with symbolic embroidery",
                typical_colors=["bright_colors", "earth_tones"],
                traditional_patterns=["symbolic_embroidery", "geometric"],
                historical_context="Traditional garment popularized in 1960s civil rights movement",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["various_west_african_styles"]
            ),
            CulturalItem(
                name="boubou",
                culture="west_african",
                region="sahel_region",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["religious_ceremony", "formal_event", "cultural_celebration"],
                inappropriate_contexts=["casual_fashion", "inappropriate_styling"],
                description="Traditional flowing robe worn across West Africa",
                typical_colors=["white", "blue", "earth_tones"],
                traditional_patterns=["embroidered", "geometric", "calligraphy"],
                historical_context="Traditional garment across multiple West African cultures",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["senegalese", "malian", "nigerian"]
            )
        ]
        
        for item in african_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_indigenous_cultural_items(self):
        """Load Indigenous cultural items"""
        indigenous_items = [
            CulturalItem(
                name="headdress",
                culture="native_american",
                region="north_america",
                significance="extremely_high",
                sacred_level="extremely_sacred",
                appropriate_contexts=["ceremonial_use_by_tribal_members"],
                inappropriate_contexts=["fashion", "costume", "commercial_use", "non_native_wear"],
                description="Sacred ceremonial item with spiritual significance",
                typical_colors=["natural", "earth_tones", "symbolic_colors"],
                traditional_patterns=["feather_arrangements", "sacred_symbols"],
                historical_context="Sacred item earned through specific tribal achievements",
                modern_adaptations_acceptable=False,
                expert_consultation_required=True,
                regional_variations=["plains_style", "woodland_style", "various_tribal_styles"]
            ),
            CulturalItem(
                name="ribbon_skirt",
                culture="native_american",
                region="north_america",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["powwow", "cultural_ceremony", "cultural_celebration"],
                inappropriate_contexts=["fashion_appropriation", "costume"],
                description="Traditional skirt with cultural and spiritual significance",
                typical_colors=["various_bright_colors"],
                traditional_patterns=["ribbon_work", "floral", "geometric"],
                historical_context="Traditional garment with tribal variations",
                modern_adaptations_acceptable=False,
                expert_consultation_required=True,
                regional_variations=["various_tribal_styles"]
            ),
            CulturalItem(
                name="jingle_dress",
                culture="ojibwe_native_american",
                region="great_lakes_region",
                significance="extremely_high",
                sacred_level="sacred",
                appropriate_contexts=["healing_ceremony", "powwow", "sacred_dance"],
                inappropriate_contexts=["fashion", "costume", "commercial_use"],
                description="Sacred healing dress with spiritual significance",
                typical_colors=["traditional_colors"],
                traditional_patterns=["jingle_cones", "sacred_designs"],
                historical_context="Originated from healing vision, sacred to Ojibwe people",
                modern_adaptations_acceptable=False,
                expert_consultation_required=True,
                regional_variations=["tribal_specific_styles"]
            )
        ]
        
        for item in indigenous_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_middle_eastern_cultural_items(self):
        """Load Middle Eastern cultural items"""
        middle_eastern_items = [
            CulturalItem(
                name="abaya",
                culture="arab",
                region="middle_east",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["religious_observance", "formal_event", "cultural_celebration"],
                inappropriate_contexts=["costume", "inappropriate_styling"],
                description="Traditional loose-fitting robe worn for modesty",
                typical_colors=["black", "navy", "earth_tones"],
                traditional_patterns=["embroidered", "geometric", "floral"],
                historical_context="Traditional garment for modesty and cultural identity",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["gulf_style", "levantine_style", "modern_cuts"]
            ),
            CulturalItem(
                name="hijab",
                culture="islamic",
                region="global",
                significance="extremely_high",
                sacred_level="sacred",
                appropriate_contexts=["religious_observance", "personal_choice"],
                inappropriate_contexts=["costume", "fashion_appropriation", "forced_wearing"],
                description="Religious head covering with spiritual significance",
                typical_colors=["various"],
                traditional_patterns=["various", "modest_designs"],
                historical_context="Religious covering mentioned in Islamic texts",
                modern_adaptations_acceptable=True,
                expert_consultation_required=True,
                regional_variations=["various_cultural_styles"]
            ),
            CulturalItem(
                name="kaftan",
                culture="ottoman_middle_eastern",
                region="middle_east_north_africa",
                significance="medium",
                sacred_level="low",
                appropriate_contexts=["formal_event", "cultural_celebration", "fashion"],
                inappropriate_contexts=["inappropriate_styling"],
                description="Traditional loose-fitting garment with regional variations",
                typical_colors=["rich_colors", "earth_tones"],
                traditional_patterns=["geometric", "floral", "embroidered"],
                historical_context="Traditional garment across Middle Eastern and North African cultures",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["moroccan", "turkish", "persian"]
            )
        ]
        
        for item in middle_eastern_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_european_cultural_items(self):
        """Load European cultural items"""
        european_items = [
            CulturalItem(
                name="dirndl",
                culture="german_austrian",
                region="central_europe",
                significance="medium",
                sacred_level="low",
                appropriate_contexts=["oktoberfest", "cultural_celebration", "formal_event"],
                inappropriate_contexts=["inappropriate_sexualization", "stereotype"],
                description="Traditional Alpine dress with cultural significance",
                typical_colors=["earth_tones", "bright_colors"],
                traditional_patterns=["floral", "checkered", "embroidered"],
                historical_context="Traditional peasant dress elevated to cultural symbol",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["bavarian", "austrian", "swiss"]
            ),
            CulturalItem(
                name="kilt",
                culture="scottish",
                region="scotland",
                significance="high",
                sacred_level="medium",
                appropriate_contexts=["scottish_celebration", "formal_event", "cultural_ceremony"],
                inappropriate_contexts=["costume", "inappropriate_styling"],
                description="Traditional Scottish garment with clan significance",
                typical_colors=["tartan_patterns"],
                traditional_patterns=["clan_tartans", "traditional_plaids"],
                historical_context="Highland dress with clan and family significance",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["highland", "lowland", "military"]
            )
        ]
        
        for item in european_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_latin_american_cultural_items(self):
        """Load Latin American cultural items"""
        latin_american_items = [
            CulturalItem(
                name="huipil",
                culture="maya_indigenous",
                region="central_america",
                significance="extremely_high",
                sacred_level="high",
                appropriate_contexts=["cultural_ceremony", "indigenous_celebration"],
                inappropriate_contexts=["fashion_appropriation", "costume", "commercial_exploitation"],
                description="Sacred Maya garment with spiritual and cultural significance",
                typical_colors=["bright_colors", "natural_dyes"],
                traditional_patterns=["maya_symbols", "sacred_geometry", "nature_motifs"],
                historical_context="Ancient Maya garment with sacred symbolism",
                modern_adaptations_acceptable=False,
                expert_consultation_required=True,
                regional_variations=["guatemalan", "mexican", "various_maya_groups"]
            ),
            CulturalItem(
                name="poncho",
                culture="andean",
                region="south_america",
                significance="medium",
                sacred_level="low",
                appropriate_contexts=["cultural_celebration", "outdoor_wear", "fashion"],
                inappropriate_contexts=["stereotype", "costume"],
                description="Traditional Andean outer garment",
                typical_colors=["earth_tones", "bright_colors"],
                traditional_patterns=["geometric", "nature_motifs", "traditional_weaves"],
                historical_context="Traditional Andean garment for protection and identity",
                modern_adaptations_acceptable=True,
                expert_consultation_required=False,
                regional_variations=["peruvian", "bolivian", "chilean"]
            )
        ]
        
        for item in latin_american_items:
            self.cultural_items[item.name] = asdict(item)
    
    def _load_educational_content(self):
        """Load educational content for cultural items"""
        self.educational_content = {
            "kimono": {
                "background": "The kimono is a traditional Japanese garment that has been worn for over 1,000 years. Originally influenced by Chinese clothing, the kimono evolved into a distinctly Japanese form of dress with deep cultural significance. Different styles of kimono are worn for different occasions, seasons, and by people of different ages and marital status.",
                "significance": "Kimono represent Japanese aesthetics, craftsmanship, and cultural values. The way a kimono is worn, its colors, patterns, and accessories all convey meaning about the wearer's status, the occasion, and the season. The art of wearing kimono (kitsuke) is considered a cultural skill.",
                "modern_context": "While not worn daily by most Japanese people today, kimono remain important for special occasions, cultural ceremonies, and as symbols of Japanese cultural identity."
            },
            "headdress": {
                "background": "Native American headdresses are sacred ceremonial items that hold deep spiritual significance. They are not fashion accessories but rather sacred regalia earned through specific achievements, spiritual visions, or tribal status. Different tribes have different styles and meanings associated with their ceremonial headwear.",
                "significance": "Each feather in a traditional headdress often represents a specific deed, honor, or spiritual achievement. The headdress connects the wearer to their ancestors, their tribal identity, and their spiritual beliefs. It is considered one of the most sacred items in Native American culture.",
                "modern_context": "The appropriation of Native American headdresses in fashion and costume contexts is deeply offensive to Native American communities and perpetuates harmful stereotypes while trivializing sacred spiritual items."
            },
            "sari": {
                "background": "The sari is an ancient garment mentioned in Indian literature dating back over 2,000 years. It consists of a long piece of unstitched cloth, typically 5-9 yards in length, that is draped around the body in various regional styles. The sari represents the rich textile traditions of the Indian subcontinent.",
                "significance": "Different regions of India have distinct draping styles, fabric choices, and occasions for wearing saris. The garment represents grace, tradition, and cultural identity. The weaving and design of saris often tell stories of local culture, mythology, and artisanal traditions.",
                "modern_context": "Saris continue to be worn regularly in India and by the Indian diaspora worldwide. While modern adaptations exist, the traditional techniques and cultural significance remain important."
            }
        }
    
    def _load_minimal_database(self):
        """Load minimal database as fallback"""
        self.logger.warning("Loading minimal cultural database")
        
        minimal_items = [
            {"name": "traditional_garment", "culture": "various", "sacred_level": "medium"},
            {"name": "ceremonial_item", "culture": "various", "sacred_level": "high"},
            {"name": "religious_clothing", "culture": "various", "sacred_level": "sacred"}
        ]
        
        for item in minimal_items:
            self.cultural_items[item["name"]] = item
    
    def get_item_info(self, item_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cultural item
        
        Args:
            item_name: Name of the cultural item
            
        Returns:
            Cultural item information or None if not found
        """
        return self.cultural_items.get(item_name.lower())
    
    def search_by_culture(self, culture: str) -> List[Dict[str, Any]]:
        """
        Search for cultural items by culture
        
        Args:
            culture: Culture name to search for
            
        Returns:
            List of cultural items from the specified culture
        """
        culture_lower = culture.lower()
        return [
            item for item in self.cultural_items.values()
            if culture_lower in item.get('culture', '').lower()
        ]
    
    def search_by_sacred_level(self, sacred_level: str) -> List[Dict[str, Any]]:
        """
        Search for cultural items by sacred level
        
        Args:
            sacred_level: Sacred level to search for
            
        Returns:
            List of cultural items with the specified sacred level
        """
        return [
            item for item in self.cultural_items.values()
            if item.get('sacred_level') == sacred_level
        ]
    
    def search_by_context(self, context: str) -> List[Dict[str, Any]]:
        """
        Search for cultural items by appropriate context
        
        Args:
            context: Context to search for
            
        Returns:
            List of cultural items appropriate for the context
        """
        context_lower = context.lower()
        return [
            item for item in self.cultural_items.values()
            if any(context_lower in ctx.lower() for ctx in item.get('appropriate_contexts', []))
        ]
    
    def find_matches(self, cultural_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find cultural items matching analysis results
        
        Args:
            cultural_analysis: Analysis results from image processing
            
        Returns:
            List of potential cultural item matches
        """
        try:
            matches = []
            
            # For each cultural item, calculate match score
            for item_name, item_data in self.cultural_items.items():
                match_score = self._calculate_analysis_match(cultural_analysis, item_data)
                if match_score > 0.5:  # Minimum match threshold
                    item_copy = item_data.copy()
                    item_copy['match_score'] = match_score
                    matches.append(item_copy)
            
            # Sort by match score
            matches.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            return matches
            
        except Exception as e:
            self.logger.error(f"Cultural matching failed: {str(e)}")
            return []
    
    def _calculate_analysis_match(self, analysis: Dict[str, Any], item: Dict[str, Any]) -> float:
        """Calculate match score between analysis and cultural item"""
        try:
            score = 0.0
            
            # Color matching
            analysis_colors = analysis.get('colors', [])
            item_colors = item.get('typical_colors', [])
            if analysis_colors and item_colors:
                color_matches = sum(1 for color in item_colors if any(
                    color.lower() in str(ac).lower() for ac in analysis_colors
                ))
                score += (color_matches / len(item_colors)) * 0.4
            
            # Pattern matching (placeholder)
            analysis_patterns = analysis.get('patterns', {})
            item_patterns = item.get('traditional_patterns', [])
            if analysis_patterns and item_patterns:
                score += 0.3  # Placeholder pattern matching
            
            # Shape/garment type matching (placeholder)
            if analysis.get('shapes'):
                score += 0.3  # Placeholder shape matching
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Match calculation failed: {str(e)}")
            return 0.0
    
    def get_educational_info(self, item_name: str) -> Dict[str, Any]:
        """
        Get educational information about a cultural item
        
        Args:
            item_name: Name of the cultural item
            
        Returns:
            Educational information about the item
        """
        return self.educational_content.get(item_name.lower(), {
            "background": "Educational information not available for this item.",
            "significance": "Cultural significance information not available.",
            "modern_context": "Modern context information not available."
        })
    
    def get_regional_info(self, garment: str, region: str) -> Dict[str, Any]:
        """
        Get regional variation information
        
        Args:
            garment: Garment name
            region: Region name
            
        Returns:
            Regional information about the garment
        """
        item_info = self.get_item_info(garment)
        if not item_info:
            return {"error": "Garment not found"}
        
        regional_variations = item_info.get('regional_variations', [])
        
        # Find region-specific information
        region_specific = None
        for variation in regional_variations:
            if region.lower() in variation.lower():
                region_specific = variation
                break
        
        if region_specific:
            return {
                'regional_name': region_specific,
                'cultural_significance': item_info.get('significance', 'unknown'),
                'appropriate_contexts': item_info.get('appropriate_contexts', []),
                'regional_variations': regional_variations,
                'modern_adaptations': item_info.get('modern_adaptations_acceptable', False)
            }
        else:
            return {
                'regional_name': f"{region}_{garment}",
                'cultural_significance': 'unknown',
                'appropriate_contexts': [],
                'regional_variations': regional_variations,
                'modern_adaptations': False
            }
    
    def add_cultural_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add new cultural item to database
        
        Args:
            item_data: Cultural item data
            
        Returns:
            Result of addition operation
        """
        try:
            # Validate required fields
            required_fields = ['name', 'culture', 'sacred_level', 'significance']
            if not all(field in item_data for field in required_fields):
                return {"success": False, "error": "Missing required fields"}
            
            # Add to database
            item_name = item_data['name'].lower()
            self.cultural_items[item_name] = item_data
            
            self.logger.info(f"Added cultural item: {item_name}")
            return {"success": True, "item_name": item_name}
            
        except Exception as e:
            self.logger.error(f"Failed to add cultural item: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def update_cultural_item(self, item_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing cultural item
        
        Args:
            item_name: Name of item to update
            updates: Fields to update
            
        Returns:
            Result of update operation
        """
        try:
            item_name_lower = item_name.lower()
            if item_name_lower not in self.cultural_items:
                return {"success": False, "error": "Item not found"}
            
            # Update item
            self.cultural_items[item_name_lower].update(updates)
            
            self.logger.info(f"Updated cultural item: {item_name_lower}")
            return {"success": True, "item_name": item_name_lower}
            
        except Exception as e:
            self.logger.error(f"Failed to update cultural item: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Database statistics and metrics
        """
        try:
            # Count items by significance level
            significance_counts = {}
            sacred_counts = {}
            culture_counts = {}
            
            for item in self.cultural_items.values():
                # Significance levels
                significance = item.get('significance', 'unknown')
                significance_counts[significance] = significance_counts.get(significance, 0) + 1
                
                # Sacred levels
                sacred_level = item.get('sacred_level', 'unknown')
                sacred_counts[sacred_level] = sacred_counts.get(sacred_level, 0) + 1
                
                # Cultures
                culture = item.get('culture', 'unknown')
                culture_counts[culture] = culture_counts.get(culture, 0) + 1
            
            # Calculate sacred items count
            sacred_items_count = (
                sacred_counts.get('sacred', 0) + 
                sacred_counts.get('extremely_sacred', 0) +
                sacred_counts.get('high', 0)
            )
            
            stats = {
                'total_items': len(self.cultural_items),
                'cultures_count': len(culture_counts),
                'sacred_items': sacred_items_count,
                'regional_variations': sum(
                    len(item.get('regional_variations', [])) 
                    for item in self.cultural_items.values()
                ),
                'significance_breakdown': significance_counts,
                'sacred_level_breakdown': sacred_counts,
                'culture_breakdown': culture_counts,
                'educational_content_items': len(self.educational_content)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Database stats calculation failed: {str(e)}")
            return {"error": str(e)}
    
    def export_database(self) -> Dict[str, Any]:
        """Export entire database for backup or analysis"""
        return {
            'cultural_items': self.cultural_items,
            'educational_content': self.educational_content,
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
    
    def get_all_cultures(self) -> List[str]:
        """Get list of all cultures in database"""
        cultures = set()
        for item in self.cultural_items.values():
            culture = item.get('culture', '')
            if culture:
                cultures.add(culture)
        return sorted(list(cultures))
    
    def get_sacred_items(self) -> List[Dict[str, Any]]:
        """Get all sacred cultural items"""
        sacred_levels = ['sacred', 'extremely_sacred']
        return [
            item for item in self.cultural_items.values()
            if item.get('sacred_level') in sacred_levels
        ]
    
    def validate_database_integrity(self) -> Dict[str, Any]:
        """Validate database integrity and completeness"""
        try:
            issues = []
            
            # Check for required fields
            required_fields = ['name', 'culture', 'significance', 'sacred_level']
            for item_name, item in self.cultural_items.items():
                for field in required_fields:
                    if field not in item:
                        issues.append(f"Missing {field} in {item_name}")
            
            # Check for empty contexts
            for item_name, item in self.cultural_items.items():
                if not item.get('appropriate_contexts'):
                    issues.append(f"No appropriate contexts defined for {item_name}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'total_items_checked': len(self.cultural_items)
            }
            
        except Exception as e:
            self.logger.error(f"Database validation failed: {str(e)}")
            return {"valid": False, "error": str(e)}