import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class OccupancyTracker:
    """Manages live occupancy based on cumulative entry and exit crossing events."""
    
    def __init__(self):
        self.total_entries = 0
        self.total_exits = 0
        
    def update_from_crossing_events(self, events: List[Dict[str, Any]]):
        """Processes line crossing events and updates occupancy counts.
        
        Args:
            events: List of line crossing events
        """
        for event in events:
            direction = event.get("direction")
            if direction == "in":
                self.total_entries += 1
            elif direction == "out":
                self.total_exits += 1
                
        # Log if there are updates
        if events:
            logger.info(
                f"[Occupancy] Updated. Total Entries: {self.total_entries}, "
                f"Total Exits: {self.total_exits}, Current Occupancy: {self.get_occupancy()}"
            )

    def get_occupancy(self) -> int:
        """Returns the current live occupancy, ensuring it is never negative."""
        occupancy = self.total_entries - self.total_exits
        return max(0, occupancy)

    def get_counts(self) -> Tuple[int, int]:
        """Returns the raw (total_entries, total_exits) counts."""
        return self.total_entries, self.total_exits
