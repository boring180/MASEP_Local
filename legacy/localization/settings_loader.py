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
    def video_path(self):
        if 'video_path' not in self.settings:
            raise Exception("Error: 'video_path' not found in settings")
        return self.settings['video_path']
    
    @property
    def callibration_path(self):
        if 'callibration_path' not in self.settings:
            raise Exception("Error: 'callibration_path' not found in settings")
        return self.settings['callibration_path']
    
    @property
    def square_size(self):
        if 'square_size' not in self.settings:
            raise Exception("Error: 'square_size' not found in settings")
        return self.settings['square_size']
    

# Global settings instance
settings = CalibrationSettings()

def main():
    settings = CalibrationSettings()
    print(settings)

if __name__ == '__main__':
    main()