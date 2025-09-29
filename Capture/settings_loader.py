import json
import cv2

class CaptureSettings:
    def __init__(self, settings_file='setting.json'):
        """Load capture settings from JSON file"""
        try:
            with open(settings_file, 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from {settings_file}")
        
    def __str__(self):
        return json.dumps(self.settings, indent=4)
    
    @property
    def cameras(self):
        if 'cameras' not in self.settings:
            raise Exception("Error: 'cameras' not found in settings")
        return self.settings['cameras']
    
    @property
    def pattern_type(self):
        if 'pattern_type' not in self.settings:
            raise Exception("Error: 'pattern_type' not found in settings")
        return self.settings['pattern_type']
    
    @property
    def pattern_size(self):
        if 'pattern_size' not in self.settings:
            raise Exception("Error: 'pattern_size' not found in settings")
        return self.settings['pattern_size']
    
    @property
    def pattern_square_size(self):
        if 'pattern_square_size' not in self.settings:
            raise Exception("Error: 'pattern_square_size' not found in settings")
        return self.settings['pattern_square_size']
    
    @property
    def chessboard_flags(self):
        if 'chessboard_flags' not in self.settings:
            raise Exception("Error: 'chessboard_flags' not found in settings")
        return self.settings['chessboard_flags']
    
    @property
    def marker_size(self):
        if 'marker_size' not in self.settings:
            raise Exception("Error: 'marker_size' not found in settings")
        return self.settings['marker_size']
    
    @property
    def aruco_dict(self):
        if 'aruco_dict' not in self.settings:
            raise Exception("Error: 'aruco_dict' not found in settings")
        return self.settings['aruco_dict']

# Global settings instance
settings = CaptureSettings()
