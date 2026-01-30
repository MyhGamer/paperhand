#!/usr/bin/env python3
"""
Polymarket Paperhanded Positions Analyzer

Script for analyzing user trades on Polymarket and identifying paperhanded positions -
positions where winning outcomes were sold before reaching maximum value (~$1.00).
"""

import requests
import json
import os
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import glob

@dataclass
class Trade:
    """Class for storing trade information"""
    proxyWallet: str
    timestamp: int
    conditionId: str
    type: str
    size: float
    usdcSize: float
    transactionHash: str
    price: float
    asset: str
    side: str
    outcomeIndex: int
    title: str
    slug: str
    icon: str
    eventSlug: str
    outcome: str
    name: str
    pseudonym: str
    bio: str
    profileImage: str
    profileImageOptimized: str

@dataclass
class Market:
    """Class for storing market information"""
    id: str
    question: str
    conditionId: str
    icon: str
    outcomes: List[str]
    outcome_prices: List[str]

@dataclass
class PositionResult:
    """Analysis result of a position"""
    question: str
    conditionId: str
    buy_shares: float
    buy_avg_price: float
    buy_usdc: float
    sell_shares: float
    sell_avg_price: float
    sell_usdc: float
    profit_usdc: float
    profit_percentage: float
    winning_outcome: str
    user_outcome: str
    is_paperhanded: bool
    outcome_prices: List[str]
    opportunity_cost: float  # How much was lost due to paperhanding

class PolymarketAnalyzer:
    def __init__(self):
        self.markets_cache = {}
        self.base_url = "https://data-api.polymarket.com"

    def load_markets_from_raw(self):
        """Loads markets from optimized file"""
        print("Loading market data...")

        # Try to load optimized version first
        optimized_file = "markets_optimized.json"
        if os.path.exists(optimized_file):
            try:
                with open(optimized_file, 'r', encoding='utf-8') as f:
                    markets_data = json.load(f)

                for condition_id, market_data in markets_data.items():
                    self.markets_cache[condition_id] = Market(
                        id=condition_id,  # Use conditionId as id
                        question=market_data.get('question', ''),
                        conditionId=condition_id,
                        icon=market_data.get('icon', ''),
                        outcomes=market_data.get('outcomes', []),
                        outcome_prices=market_data.get('outcome_prices', [])
                    )

                print(f"Loaded {len(self.markets_cache)} markets from optimized file")
                return

            except Exception as e:
                print(f"Error loading optimized file: {e}")
                print("Falling back to raw files...")

        # Fallback to raw files if optimized fails
        markets_path = "markets_raw/*.json"
        for file_path in glob.glob(markets_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    markets = json.load(f)

                for market in markets:
                    condition_id = market.get('conditionId')
                    if condition_id and market.get('question') and market.get('outcomes'):
                        try:
                            outcomes = json.loads(market['outcomes']) if isinstance(market['outcomes'], str) else market['outcomes']
                            outcome_prices = json.loads(market['outcomePrices']) if isinstance(market['outcomePrices'], str) else market['outcomePrices']

                            self.markets_cache[condition_id] = Market(
                                id=market.get('id', ''),
                                question=market.get('question', ''),
                                conditionId=condition_id,
                                icon=market.get('icon', ''),
                                outcomes=outcomes,
                                outcome_prices=[str(p) for p in outcome_prices]
                            )
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing market {condition_id}: {e}")
                            continue

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

        print(f"Loaded {len(self.markets_cache)} markets from raw files")

    def get_user_activity(self, user_address: str, limit: int = 500, use_cache: bool = True) -> List[Trade]:
        """Gets user activity from API or cache - fetches BUY and SELL trades separately to bypass 1500 trade limit"""
        cache_file = f"user_trades_{user_address.lower()}.json"

        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                cached_time = cached_data.get('timestamp', 0)
                current_time = int(time.time())

                # If cache is fresh (less than 1 hour), use it
                if current_time - cached_time < 3600:  # 1 hour
                    trades_data = cached_data.get('trades', [])
                    trades = []
                    for item in trades_data:
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

                    print(f"Loaded {len(trades)} trades from cache")
                    return trades
                else:
                    print(f"Cache expired, updating...")

            except Exception as e:
                print(f"Error loading cache: {e}")

        # If no cache or expired, load from API
        print(f"Getting user activity for {user_address}...")
        all_trades = []

        # Fetch BUY and SELL trades separately to bypass 1500 trade limit
        for side in ['BUY', 'SELL']:
            print(f"Fetching {side} trades...")
            offset = 0
            seen_transaction_hashes = set()  # Track seen transaction hashes to avoid duplicates
            max_requests = 100  # Safety limit to prevent infinite loops
            request_count = 0
            previous_response_data = None  # Store previous response for comparison

            while True:
                request_count += 1
                if request_count > max_requests:
                    print(f"Reached maximum request limit ({max_requests}), stopping pagination for {side}")
                    break

                url = f"{self.base_url}/activity"
                params = {
                    "user": user_address,
                    "type": "TRADE",
                    "side": side,  # Filter by BUY or SELL
                    "limit": limit,
                    "offset": offset
                }

                headers = {
                    'accept': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                try:
                    response = requests.get(url, params=params, headers=headers, timeout=30)
                    response.raise_for_status()

                    data = response.json()
                    if not data:
                        print(f"No more {side} data available")
                        break

                    # Check if response is empty or contains no TRADE type items
                    trade_items = [item for item in data if item.get('type') == 'TRADE']
                    if not trade_items:
                        print(f"No {side} trade items in response, stopping")
                        break

                    trades = []
                    duplicate_count = 0
                    for item in data:
                        if item.get('type') == 'TRADE':
                            tx_hash = item.get('transactionHash', '')

                            # Skip if we've seen this transaction hash before
                            if tx_hash in seen_transaction_hashes:
                                duplicate_count += 1
                                continue

                            seen_transaction_hashes.add(tx_hash)

                            trade = Trade(
                                proxyWallet=item.get('proxyWallet', ''),
                                timestamp=item.get('timestamp', 0),
                                conditionId=item.get('conditionId', ''),
                                type=item.get('type', ''),
                                size=float(item.get('size', 0)),
                                usdcSize=float(item.get('usdcSize', 0)),
                                transactionHash=tx_hash,
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

                    # Don't break on 0 new trades - continue to check for duplicate responses

                    # Create current response fingerprint for comparison (only transaction hashes)
                    current_response_hashes = tuple(sorted([item.get('transactionHash', '') for item in data if item.get('type') == 'TRADE']))

                    # Check if current response is the same as previous
                    if previous_response_data is not None and current_response_hashes == previous_response_data:
                        print(f"Response identical to previous (offset: {offset}), stopping pagination for {side}")
                        break

                    all_trades.extend(trades)
                    if len(trades) == 0:
                        print(f"No new {side} trades (offset: {offset}, duplicates: {duplicate_count}) - continuing pagination")
                    else:
                        print(f"Retrieved {len(trades)} {side} trades (offset: {offset}, duplicates: {duplicate_count})")

                    # Store current response for next iteration
                    previous_response_data = current_response_hashes

                    offset += limit

                    # Add a small delay to avoid overwhelming the API
                    time.sleep(0.1)  # Small delay between requests

                except requests.exceptions.RequestException as e:
                    print(f"API request error for {side}: {e}")
                    break

        print(f"Total retrieved {len(all_trades)} trades")

        # Save to cache
        if all_trades:
            try:
                cache_data = {
                    'timestamp': int(time.time()),
                    'user_address': user_address,
                    'trades_count': len(all_trades),
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
                        } for t in all_trades
                    ]
                }

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)

                print(f"Saved {len(all_trades)} trades to cache: {cache_file}")

            except Exception as e:
                print(f"Error saving cache: {e}")

        return all_trades

    def analyze_positions(self, trades: List[Trade]) -> List[PositionResult]:
        """Analyzes trades and identifies paperhanded positions"""
        print("Analyzing positions...")

        # Group trades by conditionId
        trades_by_market = defaultdict(list)
        for trade in trades:
            trades_by_market[trade.conditionId].append(trade)

        results = []

        for condition_id, market_trades in trades_by_market.items():
            # Sort trades by time
            market_trades.sort(key=lambda x: x.timestamp)

            # Separate buys and sells
            buys = [t for t in market_trades if t.side == 'BUY']
            sells = [t for t in market_trades if t.side == 'SELL']

            if not buys or not sells:
                continue  # Skip markets without both buys and sells

            # Calculate average prices and volumes
            buy_shares = sum(t.size for t in buys)
            buy_usdc = sum(t.usdcSize for t in buys)
            buy_avg_price = buy_usdc / buy_shares if buy_shares > 0 else 0

            sell_shares = sum(t.size for t in sells)
            sell_usdc = sum(t.usdcSize for t in sells)
            sell_avg_price = sell_usdc / sell_shares if sell_shares > 0 else 0

            profit_usdc = sell_usdc - buy_usdc
            profit_percentage = (profit_usdc / buy_usdc * 100) if buy_usdc > 0 else 0

            # Get market information
            market = self.markets_cache.get(condition_id)
            if not market:
                continue

            # Determine winning outcome
            winning_outcome = self._get_winning_outcome(market)

            # Determine which outcome user bought
            user_outcome_index = buys[0].outcomeIndex
            user_outcome = market.outcomes[user_outcome_index] if user_outcome_index < len(market.outcomes) else "Unknown"

            # Determine if this is a paperhanded position
            # Paperhanded = sold winning outcome before reaching ~$1.0
            # Doesn't matter if profit or loss - key is that it was a winning outcome
            is_paperhanded = (winning_outcome == user_outcome and
                            sell_avg_price < 0.95)  # Sold before near-maximum price

            # Calculate opportunity cost - how much was lost due to early selling
            opportunity_cost = 0
            if is_paperhanded:
                # If paperhand, calculate how much could have been made by holding to $1.0
                opportunity_cost = sell_shares * (1.0 - sell_avg_price)

            result = PositionResult(
                question=market.question,
                conditionId=condition_id,
                buy_shares=buy_shares,
                buy_avg_price=buy_avg_price,
                buy_usdc=buy_usdc,
                sell_shares=sell_shares,
                sell_avg_price=sell_avg_price,
                sell_usdc=sell_usdc,
                profit_usdc=profit_usdc,
                profit_percentage=profit_percentage,
                winning_outcome=winning_outcome,
                user_outcome=user_outcome,
                is_paperhanded=is_paperhanded,
                outcome_prices=market.outcome_prices,
                opportunity_cost=opportunity_cost
            )

            results.append(result)

        return results

    def _get_winning_outcome(self, market: Market) -> str:
        """Determines the winning outcome based on outcomePrices"""
        try:
            if not market.outcome_prices or len(market.outcome_prices) != len(market.outcomes):
                return "Unknown"

            # Look for outcome with price 1 (or very close to 1)
            for i, price_str in enumerate(market.outcome_prices):
                try:
                    price = float(price_str)
                    if price >= 0.999:  # Threshold for determining winning outcome
                        return market.outcomes[i]
                except ValueError:
                    continue

            # If no price 1, look for highest price
            max_price = 0
            winning_index = 0
            for i, price_str in enumerate(market.outcome_prices):
                try:
                    price = float(price_str)
                    if price > max_price:
                        max_price = price
                        winning_index = i
                except ValueError:
                    continue

            return market.outcomes[winning_index] if max_price > 0 else "Unknown"

        except Exception as e:
            print(f"Error determining winning outcome for {market.conditionId}: {e}")
            return "Unknown"

    def print_results(self, results: List[PositionResult], show_all: bool = False):
        """Prints analysis results"""
        paperhanded_results = [r for r in results if r.is_paperhanded]

        if not paperhanded_results:
            print("\n" + "="*60)
            print("NO PAPERHANDED POSITIONS FOUND")
            print("="*60)
            return

        total_profit = sum(r.profit_usdc for r in paperhanded_results)
        total_opportunity_cost = sum(r.opportunity_cost for r in paperhanded_results)

        # Find the position with highest opportunity cost
        max_opportunity_result = max(paperhanded_results, key=lambda x: x.opportunity_cost)

        print("\n" + "="*60)
        print("PAPERHANDED POSITIONS ANALYSIS")
        print("="*60)
        print(f"Paperhanded positions: {len(paperhanded_results)} out of {len(results)} analyzed")
        print(f"Paperhanding rate: {(len(paperhanded_results) / len(results) * 100):.1f}%")
        print(f"Total profit from paperhanded positions: ${total_profit:.2f}")
        print(f"Total lost due to paperhanding: ${total_opportunity_cost:.2f}")

        print(f"\n--- PAPERHANDED POSITIONS ---")
        for result in paperhanded_results:
            potential_max = result.sell_shares * 1.0  # Maximum could have earned
            lost_opportunity = potential_max - result.sell_usdc

            print(f"\n📊 {result.question[:100]}{'...' if len(result.question) > 100 else ''}")
            print(f"   👤 Bought: {result.user_outcome} {result.buy_shares:.1f} shares @${result.buy_avg_price:.4f} (${result.buy_usdc:.2f})")
            print(f"   💰 Sold: @${result.sell_avg_price:.4f} (${result.sell_usdc:.2f})")
            print(f"   📈 Profit: ${result.profit_usdc:.2f} ({result.profit_percentage:+.1f}%)")
            print(f"   🏆 Winning outcome: {result.winning_outcome}")
            print(f"   📋 Paperhanded: Sold at ${result.sell_avg_price:.4f} instead of ~$1.000")
            print(f"   💸 Lost opportunity: ${lost_opportunity:.2f}")

        print(f"\n--- 💀 BIGGEST PAPERHAND ---")
        print(f"🔥 {max_opportunity_result.question[:120]}{'...' if len(max_opportunity_result.question) > 120 else ''}")
        print(f"   💸 LOST OPPORTUNITY: ${max_opportunity_result.opportunity_cost:.2f}")
        print(f"   💰 Sold {max_opportunity_result.sell_shares:.1f} shares at ${max_opportunity_result.sell_avg_price:.4f}")
        print(f"   🎯 Could have been worth ${max_opportunity_result.sell_shares:.1f} at $1.000")
        print(f"   📊 User outcome: {max_opportunity_result.user_outcome} (WINNER)")

        # Additional statistics
        self._print_additional_stats(paperhanded_results, results)

        print(f"\n✅ Analysis complete!")

    def _print_additional_stats(self, paperhanded_results: List[PositionResult], all_results: List[PositionResult]):
        """Print additional statistics and insights"""
        print(f"\n" + "="*60)
        print("📊 ADDITIONAL STATISTICS")
        print("="*60)

        # Overall trading stats
        total_positions = len(all_results)
        winning_positions = [r for r in all_results if r.winning_outcome == r.user_outcome]
        losing_positions = [r for r in all_results if r.winning_outcome != r.user_outcome]

        print(f"📈 Overall Performance:")
        print(f"   Total positions analyzed: {total_positions}")
        print(f"   Winning positions: {len(winning_positions)} ({(len(winning_positions)/total_positions*100):.1f}%)")
        print(f"   Losing positions: {len(losing_positions)} ({(len(losing_positions)/total_positions*100):.1f}%)")

        # Calculate overall P&L
        total_invested = sum(r.buy_usdc for r in all_results)
        total_returned = sum(r.sell_usdc for r in all_results)
        total_pnl = total_returned - total_invested

        print(f"   Total invested: ${total_invested:.2f}")
        print(f"   Total returned: ${total_returned:.2f}")
        print(f"   Overall P&L: ${total_pnl:.2f} ({(total_pnl/total_invested*100):+.1f}%)")

        # Paperhanded specific stats
        print(f"\n🔥 Paperhanded Analysis:")

        total_paperhanded_invested = sum(r.buy_usdc for r in paperhanded_results)
        total_paperhanded_returned = sum(r.sell_usdc for r in paperhanded_results)
        total_potential_value = sum(r.sell_shares * 1.0 for r in paperhanded_results)

        print(f"   Total paperhanded invested: ${total_paperhanded_invested:.2f}")
        print(f"   Total paperhanded returned: ${total_paperhanded_returned:.2f}")
        print(f"   Potential value at $1.00: ${total_potential_value:.2f}")
        print(f"   Actual P&L from paperhands: ${(total_paperhanded_returned - total_paperhanded_invested):.2f}")
        print(f"   Maximum possible P&L: ${(total_potential_value - total_paperhanded_invested):.2f}")
        print(f"   Efficiency rate: {((total_paperhanded_returned - total_paperhanded_invested) / (total_potential_value - total_paperhanded_invested) * 100):.1f}%")

        # Worst paperhands by percentage
        worst_by_percentage = sorted(paperhanded_results, key=lambda x: x.sell_avg_price)[:3]
        print(f"\n💀 Worst Paperhands (by sale price):")
        for i, result in enumerate(worst_by_percentage, 1):
            print(f"   {i}. Sold at ${result.sell_avg_price:.4f} ({result.question[:60]}{'...' if len(result.question) > 60 else ''})")

        # Best paperhands (closest to $1.00)
        best_paperhands = sorted(paperhanded_results, key=lambda x: x.sell_avg_price, reverse=True)[:3]
        print(f"\n😅 Best Paperhands (closest to $1.00):")
        for i, result in enumerate(best_paperhands, 1):
            print(f"   {i}. Sold at ${result.sell_avg_price:.4f} ({result.question[:60]}{'...' if len(result.question) > 60 else ''})")

        # Volume analysis
        avg_sell_price = sum(r.sell_avg_price for r in paperhanded_results) / len(paperhanded_results)
        max_opportunity = max(r.opportunity_cost for r in paperhanded_results)

        print(f"\n📊 Trading Patterns:")
        print(f"   Average paperhand sell price: ${avg_sell_price:.4f}")
        print(f"   Maximum single opportunity cost: ${max_opportunity:.2f}")
        print(f"   Average opportunity cost per position: ${sum(r.opportunity_cost for r in paperhanded_results) / len(paperhanded_results):.2f}")

        # Impact on overall performance
        non_paperhanded_results = [r for r in all_results if not r.is_paperhanded]
        if non_paperhanded_results:
            non_paperhanded_pnl = sum(r.profit_usdc for r in non_paperhanded_results)
            paperhanded_pnl = sum(r.profit_usdc for r in paperhanded_results)
            total_opp_cost = sum(r.opportunity_cost for r in paperhanded_results)
            print(f"\n💡 What If Analysis:")
            print(f"   Non-paperhanded positions P&L: ${non_paperhanded_pnl:.2f}")
            print(f"   Paperhanded positions P&L: ${paperhanded_pnl:.2f}")
            print(f"   Total without paperhanding: ${(non_paperhanded_pnl + total_potential_value - total_paperhanded_invested):.2f}")
            print(f"   Performance improvement potential: ${total_opp_cost:.2f}")

def main():
    """Main function"""
    print("🎯 Polymarket Paperhanded Positions Analyzer")
    print("="*50)

    # Parse command line arguments
    user_address = None
    use_cache = True

    # Example usage: python paperhands_analyzer.py 0x22164b80e82339205F4660c519B5bD7eA8E74aCc --no-cache
    if len(sys.argv) > 1:
        user_address = sys.argv[1]
        use_cache = "--no-cache" not in sys.argv
    else:
        # Test address for automatic mode
        user_address = "0x9521e8f7a8d4b5329297f3e0ef745ff9363b766b"

    # Handle the case where only --no-cache is provided
    if user_address == "--no-cache":
        user_address = "0x9521e8f7a8d4b5329297f3e0ef745ff9363b766b"
        use_cache = False

    print(f"User address: {user_address}")
    print(f"Using cache: {'Yes' if use_cache else 'No'}")

    # Initialize analyzer
    analyzer = PolymarketAnalyzer()

    # Load market data
    analyzer.load_markets_from_raw()

    # Get user activity
    trades = analyzer.get_user_activity(user_address, use_cache=use_cache)

    if not trades:
        print("No trades found for this user")
        return

    # Analyze positions
    results = analyzer.analyze_positions(trades)

    if not results:
        print("No completed positions found (with both buys and sells)")
        return

    # Print results (only paperhanded positions)
    analyzer.print_results(results)

if __name__ == "__main__":
    main()