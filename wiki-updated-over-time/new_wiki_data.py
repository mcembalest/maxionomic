import requests
from datetime import datetime
import os
import mwclient

def get_recent_wikipedia_pages(limit=10):
    """
    Fetch the most recently created Wikipedia pages using the MediaWiki API.
    
    Args:
        limit (int): Number of recent pages to retrieve (default: 10)
    
    Returns:
        list: List of dictionaries containing page information
    """
    # API endpoint URL
    api_url = "https://en.wikipedia.org/w/api.php"
    
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rctype": "new",  # Only get new page creations
        "rclimit": limit,
        "rcnamespace": 0,  # Only get main namespace (articles)
        "rcprop": "title|timestamp|user|ids|comment"
    }
    
    try:
        # Make the API request
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process and format the results
        recent_pages = []
        for page in data["query"]["recentchanges"]:
            page_info = {
                "title": page["title"],
                "created": datetime.strptime(page["timestamp"], "%Y-%m-%dT%H:%M:%SZ"),
                "author": page["user"],
                "page_id": page["pageid"],
                "comment": page.get("comment", "No comment provided")
            }
            recent_pages.append(page_info)
        
        return recent_pages
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing API response: {e}")
        return None

def display_pages(pages):
    """
    Display the formatted page information.
    
    Args:
        pages (list): List of page dictionaries to display
    """
    if not pages:
        print("No pages found or error occurred.")
        return
        
    print("\nMost Recent Wikipedia Pages:")
    print("-" * 80)
    
    for i, page in enumerate(pages, 1):
        created_str = page["created"].strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"{i}. {page['title']}")
        print(f"   Created: {created_str}")
        print(f"   Author: {page['author']}")
        print(f"   Page ID: {page['page_id']}")
        print(f"   Comment: {page['comment']}")
        print("-" * 80)


def download_wikipedia_pages(pages, output_dir="downloaded_articles"):
    """
    Download Wikipedia pages as clean text files to a local directory,
    following redirects to get actual content.
    """
    if not pages:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    site = mwclient.Site('en.wikipedia.org')
    
    for page in pages:
        try:
            # Get the page and resolve redirects
            wiki_page = site.pages[page['title']]
            
            # Follow redirect if necessary
            if wiki_page.redirect:
                # Get the target page title from the redirect
                redirect_target = wiki_page.resolve_redirect()
                if redirect_target:
                    wiki_page = redirect_target
                    print(f"Following redirect: {page['title']} -> {wiki_page.name}")
            
            # Skip if we still don't have valid content
            if not wiki_page or not wiki_page.exists:
                print(f"Skipping {page['title']}: Page does not exist")
                continue
                
            content = wiki_page.text()
            
            # Skip if content is empty or still a redirect
            if not content or content.strip().upper().startswith('#REDIRECT'):
                print(f"Skipping {page['title']}: No content or unresolved redirect")
                continue
            
            safe_filename = f"{page['page_id']}_{page['title'].replace('/', '_')}.txt"
            file_path = os.path.join(output_dir, safe_filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"Downloaded: {page['title']} -> {file_path}")
            
        except Exception as e:
            print(f"Error downloading {page['title']}: {e}")


if __name__ == "__main__":
    # Get and display the 10 most recent pages
    recent_pages = get_recent_wikipedia_pages()
    display_pages(recent_pages)
    download_wikipedia_pages(recent_pages)