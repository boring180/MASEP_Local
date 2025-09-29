import json

class CalibrationSettings:
    def __init__(self, settings_file='setting.json'):
        """Load calibration settings from JSON file"""
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
    def center_camera(self):
        if 'center_camera' not in self.settings:
            raise Exception("Error: 'center_camera' not found in settings")
        return self.settings['center_camera']
    
    @property
    def internal_callibration_type(self):
        if 'internal_callibration_type' not in self.settings:
            raise Exception("Error: 'internal_callibration_type' not found in settings")
        return self.settings['internal_callibration_type']
    
    @property
    def pattern_internal(self):
        if 'pattern_internal' not in self.settings:
            raise Exception("Error: 'pattern_internal' not found in settings")
        return self.settings['pattern_internal']
    
    @property
    def pattern_external(self):
        if 'pattern_external' not in self.settings:
            raise Exception("Error: 'pattern_external' not found in settings")
        return self.settings['pattern_external']
    
    @property
    def pattern_size_internal(self):
        if 'pattern_size_internal' not in self.settings:
            raise Exception("Error: 'pattern_size_internal' not found in settings")
        return self.settings['pattern_size_internal']
    
    @property
    def pattern_size_external(self):
        if 'pattern_size_external' not in self.settings:
            raise Exception("Error: 'pattern_size_external' not found in settings")
        return self.settings['pattern_size_external']
    
    @property
    def pattern_square_size_internal(self):
        if 'pattern_square_size_internal' not in self.settings:
            raise Exception("Error: 'pattern_square_size_internal' not found in settings")
        return self.settings['pattern_square_size_internal']
    
    @property
    def pattern_square_size_external(self):
        if 'pattern_square_size_external' not in self.settings:
            raise Exception("Error: 'pattern_square_size_external' not found in settings")
        return self.settings['pattern_square_size_external']
    
    @property
    def same_pattern(self):
        if 'same_pattern' not in self.settings:
            raise Exception("Error: 'same_pattern' not found in settings")
        return self.settings['same_pattern']
    
    @property
    def marker_size_internal(self):
        if 'marker_size_internal' not in self.settings:
            raise Exception("Error: 'marker_size_internal' not found in settings")
        return self.settings['marker_size_internal']
    
    @property
    def marker_size_external(self):
        if 'marker_size_external' not in self.settings:
            raise Exception("Error: 'marker_size_external' not found in settings")
        return self.settings['marker_size_external']
    
    @property
    def aruco_dict_internal(self):
        if 'aruco_dict_internal' not in self.settings:
            raise Exception("Error: 'aruco_dict_internal' not found in settings")
        return self.settings['aruco_dict_internal']
    
    @property
    def aruco_dict_external(self):
        if 'aruco_dict_external' not in self.settings:
            raise Exception("Error: 'aruco_dict_external' not found in settings")
        return self.settings['aruco_dict_external']

# Global settings instance
settings = CalibrationSettings()

def main():
    settings = CalibrationSettings()
    with open('results/settings.json', 'w') as f:
        json.dump(settings.settings, f, indent=4)

if __name__ == '__main__':
    main()