
import os
import aiohttp
import asyncio
import logging
from duckduckgo_search import ddg_images

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("image_downloader.log")]
)

MAX_CONCURRENT_DOWNLOADS = 100  # Limit the number of concurrent downloads

def filter_relevant_images(results, prompt):
    """
    Filter search results to ensure relevance based on the prompt.
    """
    prompt_keywords = set(prompt.lower().split())
    filtered_results = []

    for result in results:
        title = result.get("title", "").lower()
        url = result.get("image", "").lower()
        if any(keyword in title or keyword in url for keyword in prompt_keywords):
            filtered_results.append(result["image"])
        else:
            logging.warning(f"Filtered out irrelevant image: {result['image']} (Title: {title})")
    
    return filtered_results

async def fetch_image_urls(prompt, target_count):
    """
    Fetch image URLs and filter results to ensure relevance.
    """
    urls = []
    fetched_count = 0
    while fetched_count < target_count:
        logging.info(f"Fetching image URLs for prompt: {prompt}")
        try:
            results = ddg_images(prompt, max_results=50)  # Fetch in smaller batches
            if results:
                relevant_urls = filter_relevant_images(results, prompt)
                urls.extend(relevant_urls)
                fetched_count += len(relevant_urls)
                logging.info(f"Fetched {len(relevant_urls)} relevant image URLs. Total: {len(urls)}/{target_count}")
            else:
                logging.warning("No more results found.")
                break
        except Exception as e:
            logging.error(f"Error during fetching: {e}")
            break
    return urls[:target_count]

async def download_image(session, url, save_path):
    """
    Download a single image with minimal retries for brute force.
    """
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                with open(save_path, "wb") as f:
                    f.write(await response.read())
                logging.info(f"Downloaded: {url} -> {save_path}")
                return True
            else:
                logging.warning(f"Failed to download {url} with status {response.status}")
    except Exception as e:
        logging.warning(f"Error downloading {url}: {e}")
    return False

async def download_images(image_urls, save_dir):
    """
    Download images asynchronously with filtering for prompt relevance.
    """
    os.makedirs(save_dir, exist_ok=True)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)) as session:
        tasks = []
        for idx, url in enumerate(image_urls):
            save_path = os.path.join(save_dir, f"image_{idx + 1}.jpg")
            tasks.append(download_image(session, url, save_path))
        results = await asyncio.gather(*tasks)
        downloaded_count = sum(results)
        logging.info(f"Downloaded {downloaded_count}/{len(image_urls)} images.")

async def main_async(prompt, target_count):
    """
    Main function to fetch and download strictly relevant images.
    """
    save_dir = os.path.join("downloads", prompt.replace(" ", "_"))
    # Fetch filtered URLs
    image_urls = await fetch_image_urls(prompt, target_count)
    logging.info(f"Fetched {len(image_urls)} image URLs. Starting downloads...")
    # Download filtered images
    await download_images(image_urls, save_dir)
    logging.info(f"Image downloads completed. Saved to: {save_dir}")

def main(prompt, target_count):
    """
    Entry point for the script.
    """
    asyncio.run(main_async(prompt, target_count))

if __name__ == "__main__":
    user_prompt = input("Enter your search prompt: ")
    target_count = int(input("Enter the number of images to download: "))
    main(user_prompt, target_count=target_count)

