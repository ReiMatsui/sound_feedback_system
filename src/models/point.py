from dataclasses import dataclass
import math

@dataclass
class Point:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point') -> float:
        """
        別の点との距離を計算
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)