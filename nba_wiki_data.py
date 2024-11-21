import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import logging
from time import sleep

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBASeasonScraper:
    def __init__(self, teams: List[str], years: List[int]):
        self.teams = teams
        self.years = years
        self.base_url = "https://en.wikipedia.org/wiki/{year}%E2%80%93{next_year}_{team}_season"
        
    def _get_next_year(self, year: int) -> str:
        """Convert year to two-digit format for URL."""
        return str((year + 1) % 100).zfill(2)
    
    def _format_team_name(self, team: str) -> str:
        """Format team name for URL."""
        return team.replace(" ", "_")
    
    def _get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse Wikipedia page content."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
            
    def _extract_narrative(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the opening narrative section."""
        try:
            # Find the first paragraph after the lead section
            content = []
            current_element = soup.find('p', class_=False)  # First non-classed paragraph
            
            while current_element and not current_element.find_parent('div', class_='toc'):
                if current_element.name == 'p':
                    text = current_element.get_text().strip()
                    if text:  # Only add non-empty paragraphs
                        content.append(text)
                current_element = current_element.find_next_sibling()
                
                # Stop if we hit a section header
                if current_element and current_element.name in ['h2', 'h3']:
                    break
                    
            return ' '.join(content) if content else None
        except Exception as e:
            logger.error(f"Error extracting narrative: {str(e)}")
            return None
            
    def _extract_record(self, text: str) -> Optional[float]:
        """Extract winning percentage from record text."""
        try:
            # Look for pattern like "43–39 (.524)" or "43-39 (.524)"
            match = re.search(r'\((\.?\d{3})\)', text)
            if match:
                # Convert to float, handling both ".524" and "524" formats
                percentage = match.group(1)
                if not percentage.startswith('.'):
                    percentage = f"0.{percentage}"
                return float(percentage)
            return None
        except Exception as e:
            logger.error(f"Error extracting record percentage: {str(e)}")
            return None
        
    def _extract_placement(self, text: str) -> Optional[int]:
        """Extract numerical placement from text."""
        try:
            match = re.search(r'(\d+)[stndrh]{2}', text)
            if match:
                return int(match.group(1))
            return None
        except Exception:
            return None
            
    def _determine_playoff_results(self, soup: BeautifulSoup) -> Dict[str, bool]:
        """
        Determine playoff achievements for the season based on the Playoff finish field.
        """
        playoff_results = {
            'made_playoffs': False,
            'made_conference_finals': False,
            'made_finals': False,
            'won_championship': False
        }
        
        try:
            # Find the Playoff finish row in the table
            playoff_cell = soup.find('th', string=re.compile('Playoff finish'))
            if playoff_cell:
                finish = playoff_cell.find_next('td').get_text().strip().lower()
                
                if 'did not qualify' in finish:
                    return playoff_results
                    
                # Made playoffs at minimum
                playoff_results['made_playoffs'] = True
                
                if 'conference finals' in finish:
                    playoff_results['made_conference_finals'] = True
                    
                if 'nba finals' in finish:
                    playoff_results['made_conference_finals'] = True
                    playoff_results['made_finals'] = True
                    
                if 'nba champions' in finish:
                    playoff_results['made_conference_finals'] = True
                    playoff_results['made_finals'] = True
                    playoff_results['won_championship'] = True
                    
        except Exception as e:
            logger.error(f"Error determining playoff results: {str(e)}")
            
        return playoff_results
        
    def scrape(self) -> pd.DataFrame:
        """Scrape data for all specified teams and seasons."""
        data = []
        
        for team in self.teams:
            for year in self.years:
                logger.info(f"Scraping {year}-{int(year)+1} {team} season...")
                
                # Format URL
                formatted_team = self._format_team_name(team)
                next_year = self._get_next_year(year)
                url = self.base_url.format(year=year, next_year=next_year, team=formatted_team)
                
                # Get page content
                soup = self._get_page_content(url)
                if not soup:
                    continue
                    
                season_data = {
                    'team': team,
                    'season_start': year,
                    'season_end': year + 1
                }
                
                # Extract narrative
                season_data['narrative'] = self._extract_narrative(soup)
                
                # Extract head coach
                coach_info = soup.find(string=re.compile('Head coach'))  # Changed from 'text' to 'string'
                if coach_info and coach_info.find_next():
                    season_data['head_coach'] = coach_info.find_next().get_text().strip()
                else:
                    season_data['head_coach'] = None

                # Extract record
                record_text = soup.find('th', string=re.compile('Record')).find_parent('tr') if soup.find('th', string=re.compile('Record')) else None
                season_data['win_percentage'] = None  # Default value
                if record_text:
                    season_data['win_percentage'] = self._extract_record(record_text.get_text())

                # Extract division and conference placement
                place_text = soup.find(string=re.compile('Place'))
                if place_text:
                    place_info = place_text.find_parent('tr').get_text()
                    division_match = re.search(r'Division:\s*(\d+)', place_info)
                    conference_match = re.search(r'Conference:\s*(\d+)', place_info)
                    
                    if division_match:
                        season_data['division_rank'] = int(division_match.group(1))
                    if conference_match:
                        season_data['conference_rank'] = int(conference_match.group(1))
                
                # Extract playoff results
                playoff_results = self._determine_playoff_results(soup)
                season_data.update(playoff_results)
                
                data.append(season_data)
                
                # Be nice to Wikipedia's servers
                sleep(1)
        
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    
    teams = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls',
             'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 
             'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers',
             'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 
             'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder',
             'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 
             'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards']
    years = list(range(1946, 2024))  # From first NBA season to current
    
    scraper = NBASeasonScraper(teams, years)
    df = scraper.scrape()
    
    # Display results
    print(df.head())
    
    # Save to CSV
    df.to_csv('nba_seasons.csv', index=False)