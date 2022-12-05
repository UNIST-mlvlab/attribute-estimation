import yaml

def get_settings(setting_file='settings.yaml'):
    with open(setting_file, 'r') as f:
        return yaml.load(f)


def get_model(settings):
    model_name = settings['models']['model_name']