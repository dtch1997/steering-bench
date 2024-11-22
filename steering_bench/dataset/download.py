""" Download raw datasets """

import requests
import os

from steering_bench.utils.path import raw_dataset_dir

def download_github_directory(repo_owner, repo_name, directory_path, branch='main', output_dir=None):
    """
    Download all files from a GitHub directory.
    
    Args:
        repo_owner (str): Owner of the repository
        repo_name (str): Name of the repository
        directory_path (str): Path to the directory within the repository
        branch (str): Branch name (default: 'main')
        output_dir (str): Local directory to save files (default: current directory)
    """
    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}?ref={branch}"
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = '.'

    try:
        # Get directory contents
        response = requests.get(api_url)
        response.raise_for_status()
        contents = response.json()

        # Download each file
        for item in contents:
            if item['type'] == 'file':
                # Get file content
                file_response = requests.get(item['download_url'])
                file_response.raise_for_status()
                
                # Save file
                file_path = os.path.join(output_dir, item['name'])
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                print(f"Downloaded: {item['name']}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return False
    
    return True

def download_persona():
    """ Download MWE-Persona dataset """
    download_github_directory('anthropics', 'evals', 'persona', output_dir=raw_dataset_dir / 'persona')

def download_xrisk():
    """ Download MWE-XRisk dataset """
    download_github_directory('anthropics', 'evals', 'advanced-ai-risk/lm_generated_evals', output_dir=raw_dataset_dir / 'xrisk')