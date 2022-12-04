import yaml

def get_settings(setting_file='settings.yaml'):
    with open(setting_file, 'r') as f:
        return yaml.load(f)