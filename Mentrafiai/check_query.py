"""Check one query: independent math, Network 1 + 2 outputs, routing, and Network 3."""
import logging
import sys

Q = sys.argv[1] if len(sys.argv) > 1 else (
    "I am 23, want to invest 5000 monthly SIP for 15 years with moderate risk for 🚀 wealth creation.")


def simulate(p, months, annual, step=0.0):
    r, bal, inst = annual / 1200, 0.0, p
    for m in range(months):
        if m and m % 12 == 0:
            inst *= 1 + step / 100
        bal = (bal + inst) * (1 + r)
    return bal


def main():
    print("QUERY:", Q)
    print("\n== Independent month-by-month simulation (start-of-month, r = annual/12) ==")
    for rate in (6, 8, 10, 12):
        v = simulate(5000, 180, rate)
        print(f"  {rate}%: corpus Rs {v:,.0f}  -> {v/1e5:.2f} Lakh   today's money {v/1.06**15/1e5:.2f} Lakh")
    s = simulate(5000, 180, 12, 10)
    print(f"  12% with 10% step-up: Rs {s:,.0f} -> {s/1e5:.2f} Lakh; invested "
          f"{sum(5000*1.1**y*12 for y in range(15))/1e5:.2f} Lakh; last installment Rs {5000*1.1**14:,.0f}")

    import serve
    logging.getLogger("mentrafiai-server").setLevel(logging.WARNING)
    engine = serve.engine
    engine.load()
    parsed = engine.query_parser.parse(Q)
    print("\n== Network 1 (intent/risk classifier) + Network 2 (slot parser) ==")
    print(f"  intent={parsed.intent} ({parsed.intent_confidence:.2f})  risk={parsed.risk}  horizon_class={parsed.horizon_class}")
    print(f"  age={parsed.age}  amount={parsed.amount}  mode={parsed.mode}  horizon={parsed.horizon}  goal={parsed.goal}  "
          f"name={parsed.name}  occupation={parsed.occupation}")
    reply, funds, intent, risk = engine.process_query(Q)
    print("\n== Route chosen by serve.py:", intent, "| risk:", risk, "| fund cards:", [f.name for f in funds])
    print("\n== Reply (first 1500 chars) ==\n" + reply[:1500])


if __name__ == "__main__":
    main()
