import os


def create_path(*parts):
    # path = os.path.dirname(__file__)
    path = os.getcwd()
    for part in parts:
        path = os.path.join(path, part)
        if "." not in part and not os.path.isdir(path):
            os.makedirs(path)
    return path

def save_results_to_bucket():
    script_path = create_path('gsutil_to_bucket.sh')
    os.system('sh ' + script_path)