import re

with open('/Users/pandu/Desktop/institutional-quant-research-os/production/dashboard/api/api_server.py', 'r') as f:
    content = f.read()

# 1. Add state_lock
content = content.replace("publisher = StatePublisher()", "publisher = StatePublisher()\nstate_lock = asyncio.Lock()")

# 2. Modify calculate_real_risk to return positions instead of mutating state
content = content.replace("publisher.state['positions'] = positions_dict_list", "")
content = re.sub(
    r"return \{\n\s+'var': sanitize_float\(risk_metrics\.var, 0\.0\),\n\s+'cvar': sanitize_float\(risk_metrics\.cvar, 0\.0\),\n\s+'tail_risk': sanitize_float\(risk_metrics\.tail_risk, 0\.0\)\n\s+\}",
    r"return {\n                'var': sanitize_float(risk_metrics.var, 0.0),\n                'cvar': sanitize_float(risk_metrics.cvar, 0.0),\n                'tail_risk': sanitize_float(risk_metrics.tail_risk, 0.0)\n            }, positions_dict_list",
    content
)
content = re.sub(
    r"return \{\n\s+'var': 0\.0,\n\s+'cvar': 0\.0,\n\s+'tail_risk': 0\.0\n\s+\}",
    r"return {\n        'var': 0.0,\n        'cvar': 0.0,\n        'tail_risk': 0.0\n    }, positions_dict_list",
    content
)

# 3. Fix callers of calculate_real_risk and wrap other state assignments (around line 470-490)
repl_1 = """            async with state_lock:
                publisher.state['signals'] = signal_dicts if signal_dicts else []
        except Exception as e:
            # Fallback to empty signals if AlphaManager fails
            async with state_lock:
                publisher.state['signals'] = []
        
        # Update state with real metrics
        total_pnl_val = sanitize_float(trade_metrics.total_pnl, 0.0)
        risk_data, positions_data = await asyncio.to_thread(calculate_real_risk, trade_metrics)
        async with state_lock:
            publisher.state['nav'] = sanitize_float(250_000_000 + total_pnl_val, 250_000_000.0)
            publisher.state['daily_pnl'] = total_pnl_val
            publisher.state['pnl'] = {'daily': total_pnl_val}
            publisher.state['risk'] = risk_data
            publisher.state['positions'] = positions_data
            publisher.state['updated_at'] = datetime.now(timezone.utc).isoformat()"""

content = re.sub(
    r"            publisher\.state\['signals'\] = signal_dicts if signal_dicts else \[\]\n        except Exception as e:\n            # Fallback to empty signals if AlphaManager fails\n            publisher\.state\['signals'\] = \[\]\n        \n        # Update state with real metrics\n        total_pnl_val = sanitize_float\(trade_metrics\.total_pnl, 0\.0\)\n        publisher\.state\['nav'\] = sanitize_float\(250_000_000 \+ total_pnl_val, 250_000_000\.0\)\n        publisher\.state\['daily_pnl'\] = total_pnl_val\n        publisher\.state\['pnl'\] = \{'daily': total_pnl_val\}\n        publisher\.state\['risk'\] = await asyncio\.to_thread\(calculate_real_risk, trade_metrics\)\n        publisher\.state\['updated_at'\] = datetime\.now\(timezone\.utc\)\.isoformat\(\)",
    repl_1,
    content
)

# 4. Fix lines 1927-1930
repl_2 = """            risk_data, positions_data = await asyncio.to_thread(calculate_real_risk, trade_metrics)
            async with state_lock:
                publisher.state['nav'] = sanitize_float(250_000_000.0 + total_pnl_val, 250_000_000.0)
                publisher.state['daily_pnl'] = total_pnl_val
                publisher.state['pnl'] = {'daily': total_pnl_val}
                publisher.state['risk'] = risk_data
                publisher.state['positions'] = positions_data"""

content = re.sub(
    r"            publisher\.state\['nav'\] = sanitize_float\(250_000_000\.0 \+ total_pnl_val, 250_000_000\.0\)\n            publisher\.state\['daily_pnl'\] = total_pnl_val\n            publisher\.state\['pnl'\] = \{'daily': total_pnl_val\}\n            publisher\.state\['risk'\] = await asyncio\.to_thread\(calculate_real_risk, trade_metrics\)",
    repl_2,
    content
)

# 5. Fix indices update line 1933
content = re.sub(
    r"                publisher\.state\['indices'\] = await get_indices\(\)",
    r"                indices_data = await get_indices()\n                async with state_lock:\n                    publisher.state['indices'] = indices_data",
    content
)

# 6. Fix market_status update line 1937
content = re.sub(
    r"                publisher\.state\['market_status'\] = await get_market_status\(\)",
    r"                market_data = await get_market_status()\n                async with state_lock:\n                    publisher.state['market_status'] = market_data",
    content
)

# 7. Fix regime update lines 1950-1951
content = re.sub(
    r"                        publisher\.state\['regime'\] = regimes\.iloc\[-1\]\n                        publisher\.state\['regime_confidence'\] = sanitize_float\(conf, 0\.5\)",
    r"                        async with state_lock:\n                            publisher.state['regime'] = regimes.iloc[-1]\n                            publisher.state['regime_confidence'] = sanitize_float(conf, 0.5)",
    content
)

# 8. Fix signals initialization lines 1959-1960
content = re.sub(
    r"                if 'signals' not in publisher\.state or not publisher\.state\['signals'\]:\n                    publisher\.state\['signals'\] = \[\]",
    r"                async with state_lock:\n                    if 'signals' not in publisher.state or not publisher.state['signals']:\n                        publisher.state['signals'] = []",
    content
)

# 9. Fix signals list manipulation lines 1987-1989
content = re.sub(
    r"                    current_sigs = list\(publisher\.state\.get\('signals', \[\]\)\)\n                    current_sigs\.insert\(0, new_sig\)\n                    publisher\.state\['signals'\] = current_sigs\[:15\]",
    r"                    async with state_lock:\n                        current_sigs = list(publisher.state.get('signals', []))\n                        current_sigs.insert(0, new_sig)\n                        publisher.state['signals'] = current_sigs[:15]",
    content
)

# 10. Fix updated_at assignment line 1993
content = re.sub(
    r"            publisher\.state\['updated_at'\] = datetime\.now\(timezone\.utc\)\.isoformat\(\)",
    r"            async with state_lock:\n                publisher.state['updated_at'] = datetime.now(timezone.utc).isoformat()",
    content
)


with open('/Users/pandu/Desktop/institutional-quant-research-os/production/dashboard/api/api_server.py', 'w') as f:
    f.write(content)

