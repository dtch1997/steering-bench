import requests
from pathlib import Path
from urllib.parse import urlparse, urljoin


def generate_github_url(
    repo_owner: str,
    repo_name: str,
    branch: str,
    filepath: str = None,
    raw: bool = False,
) -> str:
    """
    Generates a GitHub URL for a repository or specific file within it.

    Args:
        repo_owner (str): Owner of the repository
        repo_name (str): Name of the repository
        branch (str): Branch name or commit hash
        filepath (str, optional): Path to file within repository
        raw (bool, optional): If True, returns raw.githubusercontent.com URL

    Returns:
        str: Generated GitHub URL
    """
    # Clean up input parameters
    repo_owner = repo_owner.strip("/")
    repo_name = repo_name.strip("/")
    branch = branch.strip("/")

    if filepath:
        filepath = filepath.strip("/")

    if raw:
        base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/refs/heads/{branch}"
        if filepath:
            return urljoin(base_url + "/", filepath)
        return base_url

    base_url = f"https://github.com/{repo_owner}/{repo_name}"

    if filepath:
        return f"{base_url}/blob/{branch}/{filepath}"
    else:
        return f"{base_url}/tree/{branch}"


def list_github_directory(
    repo_owner: str, repo_name: str, branch: str, directory: str = ""
) -> list[dict[str, str]]:
    """
    Lists all files and directories in a GitHub repository directory using the GitHub API.

    Args:
        repo_owner (str): Owner of the repository
        repo_name (str): Name of the repository
        branch (str): Branch name or commit hash
        directory (str, optional): Directory path within the repository. Defaults to root.

    Returns:
        List[Dict]: List of dictionaries containing file information with keys:
            - name: Name of the file/directory
            - path: Full path within the repository
            - type: Either 'file' or 'dir'
            - size: Size in bytes (for files only)
            - download_url: Raw content URL (for files only)

    Raises:
        requests.exceptions.RequestException: If there's an error accessing the GitHub API
        ValueError: If the repository or path is invalid
    """
    # Clean up parameters
    directory = directory.strip("/")

    # Construct API URL
    api_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory}"
    )
    if branch:
        api_url += f"?ref={branch}"

    try:
        # Send request to GitHub API
        response = requests.get(api_url)
        response.raise_for_status()

        # Parse response
        contents = response.json()

        # Handle case where response is not a list (e.g., if path is a file)
        if not isinstance(contents, list):
            raise ValueError(f"Path '{directory}' does not point to a directory")

        # Process each item
        result = []
        for item in contents:
            entry = {
                "name": item["name"],
                "path": item["path"],
                "type": "dir" if item["type"] == "dir" else "file",
                "size": item.get("size", 0) if item["type"] == "file" else None,
                "download_url": item.get("download_url"),
            }
            result.append(entry)

        return result

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error accessing GitHub API: {e}")


def download_github_file(
    github_url: str, local_filepath: str, force: bool = False
) -> tuple[bool, str]:
    """
    Downloads a file from GitHub to a specified local filepath.
    By default, skips download if the file already exists locally.

    Args:
        github_url (str): URL to the GitHub file. Can be either the web URL or raw content URL.
        local_filepath (str): Local path where the file should be saved.
        force (bool, optional): If True, downloads file even if it already exists. Defaults to False.

    Returns:
        bool: True if download was successful or file exists, False if download failed
        str: Status message indicating the result

    Raises:
        ValueError: If the URL is not a valid GitHub URL
        requests.exceptions.RequestException: If there's an error downloading the file
    """
    # Convert Path object to string if necessary
    local_filepath = str(local_filepath)
    local_path = Path(local_filepath)

    # Check if file already exists
    if local_path.exists() and not force:
        return True, "File already exists, skipping download"

    # Parse the GitHub URL
    parsed_url = urlparse(github_url)
    # Check if it's a valid GitHub URL
    if "github" not in parsed_url.netloc:
        raise ValueError("Not a valid GitHub URL")

    # Convert web URL to raw content URL if necessary
    if "raw.githubusercontent.com" not in github_url:
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/blob/", "/")
    else:
        raw_url = github_url

    try:
        # Send GET request to download the file
        response = requests.get(raw_url, stream=True)
        response.raise_for_status()

        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file to disk
        with open(local_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True, "File downloaded successfully"

    except requests.exceptions.RequestException as e:
        return False, f"Error downloading file: {e}"
