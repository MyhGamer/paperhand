#!/usr/bin/env python3
"""
Polylerts PaperHand Checker Server
Simple Flask server to serve the frontend and generate statistics images
"""

# Monkey patch gevent BEFORE importing requests and other modules
try:
    from gevent import monkey
    monkey.patch_all()
    print("✅ Gevent monkey patching applied")
except ImportError:
    print("⚠️ Gevent not found, running without async optimization")
except Exception as e:
    print(f"⚠️ Gevent monkey patching failed: {e}")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import base64
import os
import sys
import time
import json
import requests
import glob
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paperhands_analyzer import PolymarketAnalyzer, Trade, Market, PositionResult

app = Flask(__name__)
CORS(app)

# Global analyzer instance
analyzer = None

# Performance optimizations
image_cache = {}
avatar_cache = {}
MAX_CACHE_SIZE = 100

# Thread pool for async operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Rate limiting
ip_request_times = {}
MAX_REQUESTS_PER_MINUTE = 10
RATE_LIMIT_WINDOW = 60  # seconds

# Cache directories
os.makedirs('cache/avatars', exist_ok=True)
os.makedirs('cache/images', exist_ok=True)

def check_rate_limit(ip_address):
    """Simple rate limiting: max requests per minute per IP"""
    current_time = time.time()

    # Clean old entries
    if ip_address in ip_request_times:
        ip_request_times[ip_address] = [
            req_time for req_time in ip_request_times[ip_address]
            if current_time - req_time < RATE_LIMIT_WINDOW
        ]
    else:
        ip_request_times[ip_address] = []

    # Check if under limit
    if len(ip_request_times[ip_address]) >= MAX_REQUESTS_PER_MINUTE:
        return False

    # Add current request
    ip_request_times[ip_address].append(current_time)
    return True

def get_results_hash(results, profile_data):
    """Generate hash for results + profile data combination"""
    hash_data = {
        'results': [
            {
                'conditionId': r.conditionId,
                'opportunity_cost': r.opportunity_cost,
                'sell_avg_price': r.sell_avg_price,
                'question': r.question[:50]  # First 50 chars for uniqueness
            } for r in results
        ],
        'profile': {
            'name': profile_data.get('name', ''),
            'profileImage': profile_data.get('profileImage', '')
        }
    }
    hash_str = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()

def get_cached_image(results_hash):
    """Get image from cache if available"""
    cache_path = f'cache/images/{results_hash}.png'
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                img_data = f.read()
            return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
        except Exception as e:
            print(f"Error loading cached image: {e}")
    return None

def cache_image(results_hash, base64_image):
    """Save image to cache"""
    try:
        # Remove data:image/png;base64, prefix
        image_data = base64.b64decode(base64_image.split(',')[1])
        cache_path = f'cache/images/{results_hash}.png'
        with open(cache_path, 'wb') as f:
            f.write(image_data)

        # Clean old cache files if too many
        cache_files = glob.glob('cache/images/*.png')
        if len(cache_files) > MAX_CACHE_SIZE:
            cache_files.sort(key=lambda x: os.path.getctime(x))
            for old_file in cache_files[:20]:  # Delete 20 oldest
                os.remove(old_file)
                print(f"Cleaned old cache file: {old_file}")

    except Exception as e:
        print(f"Error caching image: {e}")

def get_cached_avatar(profile_image_url):
    """Get avatar from cache or download and cache it"""
    if not profile_image_url:
        return None

    # Generate safe filename from URL
    url_hash = hashlib.md5(profile_image_url.encode()).hexdigest()
    cache_path = f'cache/avatars/{url_hash}.png'

    # Check cache first
    if os.path.exists(cache_path):
        # Check if cache is fresh (less than 24 hours)
        cache_age = time.time() - os.path.getctime(cache_path)
        if cache_age < 86400:  # 24 hours
            try:
                avatar_img = Image.open(cache_path)
                return avatar_img.resize((40, 40), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error loading cached avatar: {e}")

    # Download and cache new avatar
    try:
        response = requests.get(profile_image_url, timeout=5)
        response.raise_for_status()

        avatar_img = Image.open(io.BytesIO(response.content))
        # Save to cache
        avatar_img.save(cache_path, 'PNG')

        return avatar_img.resize((40, 40), Image.Resampling.LANCZOS)

    except Exception as e:
        print(f"Error downloading avatar: {e}")
        return None

def init_analyzer():
    """Initialize the analyzer with market data"""
    global analyzer
    try:
        print("🔧 Initializing PolymarketAnalyzer...")
        analyzer = PolymarketAnalyzer()

        print("📊 Loading market data...")
        analyzer.load_markets_from_raw()

        print(f"✅ Analyzer initialized successfully with {len(analyzer.markets_cache)} markets")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_font_size(text: str, font_path: str, max_width: int, max_font_size: int = 22, min_font_size: int = 14) -> int:
    """Calculate optimal font size to fit text within max_width"""
    try:
        # Start with max font size and work down
        current_size = max_font_size

        while current_size >= min_font_size:
            try:
                font = ImageFont.truetype(font_path, current_size)

                # Get text bounding box
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]

                if text_width <= max_width:
                    return current_size
            except:
                pass  # Font loading failed, try smaller size

            current_size -= 1

        # If no size fits, return minimum size
        return min_font_size

    except:
        # If anything fails, return default size
        return max_font_size

def wrap_text_smart(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Smart text wrapping that considers word boundaries and character count"""
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        # Test if adding this word would exceed the width
        test_line = current_line + (' ' if current_line else '') + word

        try:
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                # If current line is not empty, add it to lines
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, force wrap mid-word
                    lines.append(word)
                    current_line = ''
        except:
            # Font measurement failed, use fallback method
            if len(test_line) <= 50:  # Approximate character limit
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

    # Add remaining line
    if current_line:
        lines.append(current_line)

    return lines

def get_user_profile_data(address: str):
    """Get user profile data from Polymarket API"""
    try:
        url = f"https://polymarket.com/api/profile/userData?address={address}"
        headers = {
            'accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract relevant fields
        profile_data = {
            'name': data.get('name', ''),
            'pseudonym': data.get('pseudonym', ''),
            'profileImage': data.get('profileImage', ''),
            'xUsername': data.get('xUsername', ''),
            'verifiedBadge': data.get('verifiedBadge', False),
            'displayUsernamePublic': data.get('displayUsernamePublic', False),
            'bio': data.get('bio', ''),
            'createdAt': data.get('createdAt', ''),
            'proxyWallet': data.get('proxyWallet', address)
        }

        print(f"Profile data retrieved for {address}: {profile_data.get('name', 'Unknown')}")
        return profile_data

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch profile data for {address}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse profile data for {address}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching profile data for {address}: {e}")
        return None

def get_user_activity_with_cache(address: str, use_cache: bool = True):
    """Get user activity with caching functionality similar to paperhands_analyzer.py"""
    cache_file = f"user_trades_{address.lower()}.json"

    # Try to load from cache if enabled
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            cached_time = cached_data.get('timestamp', 0)
            current_time = int(time.time())

            # If cache is fresh (less than 1 hour), use it
            if current_time - cached_time < 3600:  # 1 hour
                print(f"Loaded {len(cached_data.get('trades', []))} trades from cache")
                profile_data = cached_data.get('profile_data', {})
                trades = []
                for item in cached_data.get('trades', []):
                    trade = Trade(
                        proxyWallet=item.get('proxyWallet', ''),
                        timestamp=item.get('timestamp', 0),
                        conditionId=item.get('conditionId', ''),
                        type=item.get('type', ''),
                        size=float(item.get('size', 0)),
                        usdcSize=float(item.get('usdcSize', 0)),
                        transactionHash=item.get('transactionHash', ''),
                        price=float(item.get('price', 0)),
                        asset=item.get('asset', ''),
                        side=item.get('side', ''),
                        outcomeIndex=int(item.get('outcomeIndex', 0)),
                        title=item.get('title', ''),
                        slug=item.get('slug', ''),
                        icon=item.get('icon', ''),
                        eventSlug=item.get('eventSlug', ''),
                        outcome=item.get('outcome', ''),
                        name=item.get('name', ''),
                        pseudonym=item.get('pseudonym', ''),
                        bio=item.get('bio', ''),
                        profileImage=item.get('profileImage', ''),
                        profileImageOptimized=item.get('profileImageOptimized', '')
                    )
                    trades.append(trade)

                # Return trades along with profile data (we'll use this later)
                return trades, profile_data
            else:
                print(f"Cache expired for {address}")
        except Exception as e:
            print(f"Error loading cache for {address}: {e}")

    # If no cache or expired, get fresh data
    print(f"Getting fresh data for {address}...")
    trades = analyzer.get_user_activity(address, use_cache=False)  # Don't use internal cache

    # Also get fresh profile data
    profile_data = get_user_profile_data(address) or {}

    # Save to cache
    if trades:
        try:
            cache_data = {
                'timestamp': int(time.time()),
                'user_address': address,
                'trades_count': len(trades),
                'profile_data': profile_data,  # Add profile data to cache
                'trades': [
                    {
                        'proxyWallet': t.proxyWallet,
                        'timestamp': t.timestamp,
                        'conditionId': t.conditionId,
                        'type': t.type,
                        'size': t.size,
                        'usdcSize': t.usdcSize,
                        'transactionHash': t.transactionHash,
                        'price': t.price,
                        'asset': t.asset,
                        'side': t.side,
                        'outcomeIndex': t.outcomeIndex,
                        'title': t.title,
                        'slug': t.slug,
                        'icon': t.icon,
                        'eventSlug': t.eventSlug,
                        'outcome': t.outcome,
                        'name': t.name,
                        'pseudonym': t.pseudonym,
                        'bio': t.bio,
                        'profileImage': t.profileImage,
                        'profileImageOptimized': t.profileImageOptimized
                    } for t in trades
                ]
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(trades)} trades and profile data to cache")
        except Exception as e:
            print(f"Error saving cache for {address}: {e}")

    return trades, profile_data

def load_local_image(image_path, default_size=(40, 40)):
    """Load local image file"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img.resize(default_size, Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
    return None

def create_stats_image(results, profile_data=None):
    """Create a statistics image using PIL with local assets and user profile"""
    paperhanded_results = [r for r in results if r.is_paperhanded]

    if not paperhanded_results:
        return None

    # Load emoji images
    skull_img = load_local_image("paperhand.png", (80, 80))
    money_img = load_local_image("money.png", (60, 60))
    bag_img = load_local_image("bag.png", (16, 16))  # Small bag icon for sale info

    # Load user profile image from cache if available
    user_avatar = None
    if profile_data and profile_data.get('profileImage'):
        user_avatar = get_cached_avatar(profile_data['profileImage'])

    # Image dimensions - increased height for better spacing
    width, height = 800, 950

    # Load background image or create black background
    try:
        background_img = Image.open("Udddntitled.jpg")
        # Resize and crop to fit dimensions
        background_img = background_img.resize((width, height), Image.Resampling.LANCZOS)
        # Convert to RGB if needed
        if background_img.mode != 'RGB':
            background_img = background_img.convert('RGB')
        img = background_img.copy()
    except Exception as e:
        print(f"Failed to load background image: {e}")
        # Fallback to black background
        img = Image.new('RGB', (width, height), color='black')

    draw = ImageDraw.Draw(img)

    # Try to load custom fonts with specific font assignments
    try:
        main_font = ImageFont.truetype("Inter_18pt-Bold.ttf", 48)      # Main amounts - 48px Inter Bold
        secondary_font = ImageFont.truetype("Inter_18pt-Bold.ttf", 28) # Secondary amounts - 28px Inter Bold
        header_font = ImageFont.truetype("font.ttf", 16)              # Headers - 16px regular
        text_font = ImageFont.truetype("font.ttf", 22)                # Card text - 22px regular
        small_font = ImageFont.truetype("font.ttf", 14)               # Position badge - 14px regular
        tiny_font = ImageFont.truetype("font.ttf", 25)                # Footer - 11px regular
    except:
        try:
            # Fallback to regular font for amounts if Inter Bold not available
            main_font = ImageFont.truetype("font.ttf", 48)
            secondary_font = ImageFont.truetype("font.ttf", 28)
            header_font = ImageFont.truetype("font.ttf", 16)
            text_font = ImageFont.truetype("font.ttf", 22)
            small_font = ImageFont.truetype("font.ttf", 14)
            tiny_font = ImageFont.truetype("font.ttf", 25)
        except:
            # Use default font
            main_font = ImageFont.load_default()
            secondary_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            tiny_font = ImageFont.load_default()

    # Calculate stats
    total_opportunity_cost = sum(r.opportunity_cost for r in paperhanded_results)
    max_opportunity_result = max(paperhanded_results, key=lambda x: x.opportunity_cost)
    worst_sale_result = min(paperhanded_results, key=lambda x: x.sell_avg_price)  # Lowest sale price

    # Profile section: Top left corner - only show if we have avatar or name
    user_name = ''
    if profile_data:
        user_name = profile_data.get('name', '')
        if not user_name:
            user_name = profile_data.get('pseudonym', '')

    # Only show profile block if we have valid name OR avatar
    if (user_name and user_name != 'Unknown') or user_avatar:
        profile_box_width = 200
        profile_box_height = 60
        profile_x = 20
        profile_y = 20

        # Draw rounded rectangle for profile
        draw.rounded_rectangle([profile_x, profile_y, profile_x + profile_box_width, profile_y + profile_box_height],
                              radius=8, fill='#111111', outline='#333333', width=1)

        # Avatar or placeholder
        avatar_x = profile_x + 10
        avatar_y = profile_y + 10

        if user_avatar:
            # Create circular mask for avatar
            mask = Image.new('L', (40, 40), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse([0, 0, 40, 40], fill=255)

            # Apply circular mask to avatar
            avatar_circular = Image.new('RGBA', (40, 40), (0, 0, 0, 0))
            avatar_circular.paste(user_avatar, (0, 0))
            avatar_circular.putalpha(mask)

            img.paste(avatar_circular, (avatar_x, avatar_y), avatar_circular)

        # Only draw name if we have a valid name
        if user_name and user_name != 'Unknown':
            # Calculate font size for name to fit - increased to 21px max
            name_x = avatar_x + 50  # 10px gap after avatar
            name_width = profile_box_width - 60  # Available width for text
            optimal_name_size = calculate_font_size(user_name, "font.ttf", name_width, 21, 14)

            try:
                name_font = ImageFont.truetype("font.ttf", optimal_name_size)
            except:
                name_font = text_font  # Fallback to text_font (22px) for larger size

            # Center name vertically with avatar (perfect centering) - moved up slightly
            name_bbox = name_font.getbbox(user_name)
            name_height = name_bbox[3] - name_bbox[1]
            avatar_center_y = avatar_y + 5  # Moved up 2px from center (20 -> 18)
            name_y = avatar_center_y - name_height // 2  # Perfect vertical center

            draw.text((name_x, name_y), user_name, fill='#FFFFFF', font=name_font)


    # Main section: Centered composition with proper hierarchy
    center_x = width // 2

    # Adjust skull position if profile box is present
    skull_y_offset = 100 if (user_avatar or (profile_data and profile_data.get('name'))) else 40

    # Smaller skull emoji (main anchor)
    if skull_img:
        large_skull = skull_img.resize((70, 70), Image.Resampling.LANCZOS)  # Smaller size
        skull_x = center_x - 35
        img.paste(large_skull, (skull_x, skull_y_offset), large_skull.convert("RGBA"))

    # Calculate adjusted positions based on profile presence
    has_profile = user_avatar or (profile_data and profile_data.get('name'))
    base_y_offset = 60 if has_profile else 0  # Extra space when profile is present

    # Main title - larger
    title_text = "I LOST DUE PAPERHANDING"
    title_y = 140 + base_y_offset
    draw.text((center_x, title_y), title_text,
              fill='#999999', font=secondary_font, anchor="mm")  # Lighter gray for better visibility

    # Main amount - THE STAR (larger, just slightly bigger than title)
    main_amount_text = f"${total_opportunity_cost:.2f}"
    main_amount_y = 175 + base_y_offset
    draw.text((center_x, main_amount_y), main_amount_text,
              fill='#FF3B30', font=main_font, anchor="mm")  # Slightly closer to title

    # Secondary section: My Biggest L
    secondary_header_y = 250 + base_y_offset
    secondary_header_text = "MY BIGGEST L"
    draw.text((center_x, secondary_header_y), secondary_header_text,
              fill='#999999', font=header_font, anchor="mm")  # Lighter gray for better visibility

    # Secondary amount - less prominent
    secondary_amount_text = f"${max_opportunity_result.opportunity_cost:.2f}"
    draw.text((center_x, secondary_header_y + 35), secondary_amount_text,
              fill='#FF3B30', font=secondary_font, anchor="mm")  # Same red, smaller

    # WORST section: My Worst L
    worst_header_y = secondary_header_y + 290  # Increased spacing (40px more)
    worst_header_text = "MY WORST L"
    draw.text((center_x, worst_header_y), worst_header_text,
              fill='#999999', font=header_font, anchor="mm")  # Lighter gray for better visibility

    # Worst amount - amount lost from worst sale
    worst_amount_text = f"${worst_sale_result.opportunity_cost:.2f}"
    draw.text((center_x, worst_header_y + 35), worst_amount_text,
              fill='#FF3B30', font=secondary_font, anchor="mm")  # Same red, smaller

    # Card for market content - increased 1.5x size
    card_width = 540  # 360 * 1.5
    card_height = 126  # 84 * 1.5
    card_x = (width - card_width) // 2  # Center the card
    card_y = secondary_header_y + 80  # 24px margin from secondary section

    # Draw rounded rectangle card with precise specs
    draw.rounded_rectangle([card_x, card_y, card_x + card_width, card_y + card_height],
                          radius=16, fill='#111111', outline='#333333', width=1)

    # Get market info
    market = analyzer.markets_cache.get(max_opportunity_result.conditionId)
    market_icon_url = market.icon if market else ""

    # Load market icon
    market_icon = None
    if market_icon_url and market_icon_url.startswith('http'):
        try:
            import urllib.request
            with urllib.request.urlopen(market_icon_url, timeout=5) as response:
                icon_data = response.read()
                market_icon = Image.open(io.BytesIO(icon_data))
                market_icon = market_icon.resize((50, 50), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Failed to load market icon: {e}")

    # Card content with 1.5x scaled specifications
    padding = 24  # 16 * 1.5
    icon_size = 84  # 56 * 1.5
    gap = 18  # 12 * 1.5

    # Market icon positioned first (top-left of card content)
    icon_x = card_x + padding
    icon_y = card_y + padding

    if market_icon:
        # Resize to 84x84 (1.5x larger)
        market_icon_resized = market_icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        img.paste(market_icon_resized, (icon_x, icon_y), market_icon_resized.convert("RGBA"))

    # Title text positioned right of icon with gap
    title_x = icon_x + icon_size + gap
    title_y = card_y + padding

    # Calculate optimal font size and smart text wrapping
    question = max_opportunity_result.question
    available_width = card_width - padding * 2 - icon_size - gap  # Total available width for text

    # Calculate optimal font size for the title (between 14px and 22px)
    optimal_font_size = calculate_font_size(question, "font.ttf", available_width, 22, 14)

    try:
        title_font = ImageFont.truetype("font.ttf", optimal_font_size)
    except:
        title_font = text_font  # Fallback to default font

    # Smart text wrapping with calculated font size
    wrapped_lines = wrap_text_smart(question, title_font, available_width)

    # Draw text lines (max 2 lines)
    line_height = optimal_font_size + 8  # Dynamic line height based on font size
    max_lines = min(2, len(wrapped_lines))

    for i in range(max_lines):
        line = wrapped_lines[i]
        draw.text((title_x, title_y), line,
                  fill='#FFFFFF', font=title_font)
        title_y += line_height

    # Sale info - bottom of card with proper spacing from content (moved closer to title)
    sale_shares = int(max_opportunity_result.sell_shares)
    sale_price = max_opportunity_result.sell_avg_price
    sale_y = title_y + 12  # Reduced gap between title and sale info

    # Draw bag icon if available (smaller size)
    if bag_img:
        bag_icon_scaled = bag_img.resize((14, 14), Image.Resampling.LANCZOS)  # Smaller 14x14px
        img.paste(bag_icon_scaled, (title_x, sale_y + 2), bag_icon_scaled.convert("RGBA"))  # Better vertical alignment
        # Adjust text position after icon
        text_x = title_x + 20  # Proper spacing with smaller icon
    else:
        # Fallback to text emoji
        text_x = title_x

    sale_info_text = f"Sold {sale_shares} shares at ${sale_price:.3f}"
    draw.text((text_x, sale_y), sale_info_text,
              fill='#999999', font=small_font)  # Lighter gray for better visibility

    # Second card for WORST market content
    worst_card_y = worst_header_y + 80  # Same spacing as first card

    # Draw rounded rectangle for worst card with same specs
    draw.rounded_rectangle([card_x, worst_card_y, card_x + card_width, worst_card_y + card_height],
                          radius=16, fill='#111111', outline='#333333', width=1)

    # Get worst market info
    worst_market = analyzer.markets_cache.get(worst_sale_result.conditionId)
    worst_market_icon_url = worst_market.icon if worst_market else ""

    # Load worst market icon
    worst_market_icon = None
    if worst_market_icon_url and worst_market_icon_url.startswith('http'):
        try:
            import urllib.request
            with urllib.request.urlopen(worst_market_icon_url, timeout=5) as response:
                icon_data = response.read()
                worst_market_icon = Image.open(io.BytesIO(icon_data))
                worst_market_icon = worst_market_icon.resize((50, 50), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Failed to load worst market icon: {e}")

    # Worst card content with same 1.5x scaled layout
    padding = 24  # 16 * 1.5
    icon_size = 84  # 56 * 1.5
    gap = 18  # 12 * 1.5

    # Worst market icon
    worst_icon_x = card_x + padding
    worst_icon_y = worst_card_y + padding

    if worst_market_icon:
        worst_market_icon_resized = worst_market_icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        img.paste(worst_market_icon_resized, (worst_icon_x, worst_icon_y), worst_market_icon_resized.convert("RGBA"))

    # Worst title text
    worst_title_x = worst_icon_x + icon_size + gap
    worst_title_y = worst_card_y + padding

    # Calculate optimal font size for worst market title
    worst_question = worst_sale_result.question
    worst_available_width = card_width - padding * 2 - icon_size - gap

    # Calculate optimal font size for worst title
    worst_optimal_font_size = calculate_font_size(worst_question, "font.ttf", worst_available_width, 22, 14)

    try:
        worst_title_font = ImageFont.truetype("font.ttf", worst_optimal_font_size)
    except:
        worst_title_font = text_font  # Fallback to default font

    # Smart text wrapping for worst market
    worst_wrapped_lines = wrap_text_smart(worst_question, worst_title_font, worst_available_width)

    # Draw worst text lines (max 2 lines)
    worst_line_height = worst_optimal_font_size + 8
    worst_max_lines = min(2, len(worst_wrapped_lines))

    for i in range(worst_max_lines):
        line = worst_wrapped_lines[i]
        draw.text((worst_title_x, worst_title_y), line,
                  fill='#FFFFFF', font=worst_title_font)
        worst_title_y += worst_line_height

    # Worst sale info - bottom of second card (moved closer to title)
    worst_sale_shares = int(worst_sale_result.sell_shares)
    worst_sale_price = worst_sale_result.sell_avg_price
    worst_sale_y = worst_title_y + 12  # Reduced gap between title and sale info

    # Draw bag icon for worst card if available (smaller size)
    if bag_img:
        bag_icon_scaled = bag_img.resize((14, 14), Image.Resampling.LANCZOS)  # Smaller 14x14px
        img.paste(bag_icon_scaled, (worst_title_x, worst_sale_y + 2), bag_icon_scaled.convert("RGBA"))  # Better vertical alignment
        # Adjust text position after icon
        worst_text_x = worst_title_x + 20  # Proper spacing with smaller icon
    else:
        # Fallback to text emoji
        worst_text_x = worst_title_x

    worst_sale_info_text = f"Sold {worst_sale_shares} shares at ${worst_sale_price:.3f}"
    draw.text((worst_text_x, worst_sale_y), worst_sale_info_text,
              fill='#999999', font=small_font)  # Lighter gray for better visibility

    # Footer - barely visible with star
    footer_text = "x.com/polylerts"
    draw.text((width//2 - 20, height - 30), footer_text,
              fill='#333333', font=tiny_font, anchor="mm")  # Very dark gray, barely visible



    # Convert image to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Serve the main page"""
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze user address and return statistics image"""
    if not analyzer:
        return jsonify({'success': False, 'error': 'Analyzer not initialized'}), 500

    try:
        start_time = time.time()

        # Rate limiting check
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        if not check_rate_limit(client_ip):
            return jsonify({
                'success': False,
                'error': 'Rate limit exceeded. Please try again later.',
                'retry_after': RATE_LIMIT_WINDOW
            }), 429

        data = request.json
        address = data.get('address', '').strip()
        force_refresh = data.get('force_refresh', False)

        if not address:
            return jsonify({'success': False, 'error': 'Address required'}), 400

        print(f"🚀 Starting fast analysis for {address} (force_refresh: {force_refresh}) from {client_ip}")

        # Submit parallel tasks
        profile_future = thread_pool.submit(get_user_profile_data, address)
        trades_future = thread_pool.submit(get_user_activity_with_cache, address, not force_refresh)

        # Get results with timeout
        try:
            profile_data = profile_future.result(timeout=15)  # 15 sec timeout for profile
            trades_result = trades_future.result(timeout=30)  # 30 sec timeout for trades
        except Exception as e:
            print(f"Timeout or error in parallel requests: {e}")
            return jsonify({'success': False, 'error': 'Request timeout, please try again'}), 504

        # Handle trades result format
        if isinstance(trades_result, tuple):
            trades = trades_result[0]
            cached_profile = trades_result[1]
            # Use fetched profile if cache doesn't have it
            profile_data = profile_data or cached_profile
        else:
            trades = trades_result

        if not trades:
            return jsonify({'success': False, 'error': 'No trades found for this address'}), 404

        # Analyze positions
        results = analyzer.analyze_positions(trades)
        if not results:
            return jsonify({'success': False, 'error': 'No completed positions found'}), 404

        paperhanded_results = [r for r in results if r.is_paperhanded]
        if not paperhanded_results:
            return jsonify({'success': False, 'error': 'No paperhanded positions found'}), 404

        # Check image cache first
        results_hash = get_results_hash(paperhanded_results, profile_data or {})
        cached_image = get_cached_image(results_hash)

        if cached_image and not force_refresh:
            print(f"✅ Image loaded from cache in {time.time() - start_time:.2f}s")
            return jsonify({
                'success': True,
                'image_url': cached_image,
                'user_profile': profile_data or {},
                'stats': {
                    'total_trades': len(trades),
                    'total_positions': len(results),
                    'paperhanded_count': len(paperhanded_results),
                    'cache_used': True,
                    'generation_time': time.time() - start_time
                }
            })

        # Generate new image if not in cache
        print(f"🎨 Generating new image...")
        image_data = create_stats_image(paperhanded_results, profile_data or {})

        if not image_data:
            return jsonify({'success': False, 'error': 'Failed to generate image'}), 500

        # Cache the generated image
        cache_image(results_hash, image_data)

        total_time = time.time() - start_time
        print(f"✅ Analysis completed in {total_time:.2f}s")

        return jsonify({
            'success': True,
            'image_url': image_data,
            'user_profile': profile_data or {},
            'stats': {
                'total_trades': len(trades),
                'total_positions': len(results),
                'paperhanded_count': len(paperhanded_results),
                'cache_used': False,
                'generation_time': total_time
            }
        })

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'analyzer_initialized': analyzer is not None})

@app.route('/clear_cache/<address>', methods=['DELETE'])
def clear_cache(address):
    """Clear cache for specific address"""
    try:
        cache_file = f"user_trades_{address.lower()}.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            return jsonify({'success': True, 'message': f'Cache cleared for {address}'})
        else:
            return jsonify({'success': False, 'error': 'No cache found for this address'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Check if running with Gunicorn
    if 'gunicorn' in os.environ.get('SERVER_SOFTWARE', '').lower():
        # Production mode with Gunicorn
        print("🚀 Starting with Gunicorn in production mode...")

        # Initialize analyzer
        if not init_analyzer():
            print("❌ Failed to initialize analyzer. Exiting.")
            sys.exit(1)

        print("✅ Analyzer initialized successfully!")
        print("📊 Ready to analyze paperhanded positions!")
    else:
        # Development mode with Flask dev server
        print("🚀 Starting Polymarket PaperHand Checker Server (Development)...")

        # Initialize analyzer
        if not init_analyzer():
            print("❌ Failed to initialize analyzer. Exiting.")
            sys.exit(1)

        print("✅ Analyzer initialized successfully!")
        print("🌐 Server starting on http://localhost:5000")
        print("📊 Ready to analyze paperhanded positions!")
        print("💡 Use 'start_production.sh' for production deployment")

        app.run(host='0.0.0.0', port=5000, debug=False)