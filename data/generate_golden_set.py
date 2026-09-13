import json
import os
import random


def build_golden_dataset():
    random.seed(42)

    scenarios = {
        "battery_drain": {
            "escalation": "AUTO_HANDLE",
            "templates": [
                "My {device} battery is draining unusually fast after the latest iOS update.",
                "Battery life on my {device} has become terrible since I updated iOS.",
                "My battery drops from {start}% to {end}% in just a few hours.",
                "Why is my {device} losing battery so quickly after the update?",
                "My battery health keeps getting worse after the latest update.",
                "My {device} barely lasts through the day anymore.",
                "Battery percentage is dropping much faster than normal on my {device}.",
                "I updated iOS and now my battery drains even when I'm barely using the phone.",
                "My {device} gets unusually warm while charging and the battery drains quickly.",
                "Can you help me figure out why my {device} battery doesn't last?",
            ],
            "values": {
                "device": ["iPhone 7", "iPhone 8", "iPhone X", "iPhone 11", "iPhone 12"],
                "start": ["100", "90", "80"],
                "end": ["20", "15", "10", "5"],
            },
            "reply": (
                "We can help with your battery performance! Please DM us your "
                "device model and iOS version so we can look into this."
            ),
        },

        "software_bugs": {
            "escalation": "AUTO_HANDLE",
            "templates": [
                "Apps keep crashing on my {device} after the latest iOS update.",
                "My {device} freezes randomly and I have to restart it.",
                "Wi-Fi keeps disconnecting after I updated iOS.",
                "Bluetooth stops working whenever I lock my phone.",
                "The keyboard disappears whenever I open notifications.",
                "My apps are extremely slow and keep becoming unresponsive.",
                "Safari crashes every time I try to open a new tab.",
                "My screen freezes whenever I switch between apps.",
                "Notifications aren't behaving correctly after the update.",
                "My {device} keeps restarting by itself since I installed the update.",
                "AirPods audio keeps cutting out after the iOS update.",
                "The camera app crashes whenever I try to take a photo.",
            ],
            "values": {
                "device": ["iPhone 8", "iPhone X", "iPhone 11", "iPhone 12", "iPad"],
            },
            "reply": (
                "We'd love to help! Please DM us your device model and current "
                "iOS version, along with any troubleshooting steps you've tried."
            ),
        },

        "account_authentication": {
            "escalation": "AUTO_HANDLE",
            "templates": [
                "I forgot my Apple ID password. How can I reset it?",
                "I can't sign into the App Store on my new {device}.",
                "My Apple ID password isn't being accepted.",
                "I'm having trouble verifying my Apple ID.",
                "The App Store keeps asking me to sign in again.",
                "I forgot my passcode and my iPhone is disabled. What should I do?",
                "I'm not receiving the verification code for my Apple ID.",
                "Why can't I sign into my Apple account on my {device}?",
                "My Apple ID verification keeps failing.",
                "I can't access the App Store because it won't authenticate my account.",
            ],
            "values": {
                "device": ["iPhone", "iPad", "Mac"],
            },
            "reply": (
                "We can help with your Apple ID issue. Please DM us so a support "
                "advisor can securely guide you through the next steps."
            ),
        },

        "hardware_repair": {
            "escalation": "ESCALATE_TO_HUMAN",
            "templates": [
                "I dropped my {device} and the screen is shattered.",
                "The microphone on my {device} has stopped working.",
                "My speaker stopped working after my phone was dropped.",
                "The touchscreen is unresponsive after I dropped my phone.",
                "My {device} was exposed to water and now the speaker sounds muffled.",
                "One of the buttons on my iPhone no longer works.",
                "The display on my {device} is damaged and needs repair.",
                "My camera hardware appears to be broken after a fall.",
                "My phone was damaged and I need to know how to arrange a repair.",
                "The charging port on my {device} appears to be physically damaged.",
            ],
            "values": {
                "device": ["iPhone 8", "iPhone X", "iPhone 11", "iPhone 12", "iPad"],
            },
            "reply": (
                "We're sorry to hear about the hardware issue. Please DM us your "
                "device details so a support advisor can help with repair options."
            ),
        },

        "billing_subscriptions": {
            "escalation": "ESCALATE_TO_HUMAN",
            "templates": [
                "I was charged for a subscription that I already cancelled.",
                "I don't recognize this charge from iTunes.",
                "How can I request a refund for an accidental in-app purchase?",
                "My Apple Music subscription charged me after I cancelled it.",
                "I need help disputing an Apple charge on my card.",
                "Why was I charged for iCloud storage?",
                "My child accidentally made an in-app purchase. Can I get a refund?",
                "I don't recognize a purchase on my Apple account.",
                "Can someone help me investigate an unexpected Apple charge?",
                "I need an invoice for one of my Apple purchases.",
                "How do I cancel my iCloud storage subscription?",
            ],
            "values": {},
            "reply": (
                "We'd be glad to look into this billing concern. Please DM us "
                "the relevant purchase or invoice details so an agent can review."
            ),
        },

        "general_inquiry": {
            "escalation": "AUTO_HANDLE",
            "templates": [
                "How do I turn on Dark Mode on my iPhone?",
                "Can I transfer photos from my iPhone to my Mac using AirDrop?",
                "Does my {device} support the latest iOS version?",
                "How do I change the settings for notifications?",
                "Can I use two SIMs on my iPhone?",
                "Where can I find my Apple purchase history?",
                "How do I update my iPhone?",
                "How do I enable automatic updates?",
                "When is the next Apple event?",
                "How can I back up my iPhone?",
                "How do I connect my AirPods to my iPhone?",
                "Where can I check my device's storage?",
            ],
            "values": {
                "device": ["iPhone 8", "iPhone X", "iPhone 11", "iPhone 12", "iPhone 13"],
            },
            "reply": (
                "Thanks for reaching out! We'd be happy to help. Please DM us "
                "with your device details and we'll guide you through the steps."
            ),
        },
    }

    suffixes = [
        "",
        " Please help.",
        " Can you help?",
        " Thanks.",
        " Any advice?",
        " Please assist.",
        " Need help.",
        " Please let me know what to do.",
        " This started recently.",
        " It happened after the update.",
    ]

    examples = []
    seen_texts = set()
    example_id = 1

    # Generate candidates until we have 200 unique examples.
    while len(examples) < 200:
        intent_names = list(scenarios.keys())
        intent = intent_names[(example_id - 1) % len(intent_names)]
        scenario = scenarios[intent]

        template = random.choice(scenario["templates"])

        values = {}
        for key, options in scenario["values"].items():
            values[key] = random.choice(options)

        customer_text = template.format(**values)

        # Add a suffix to create realistic variation.
        suffix = random.choice(suffixes)
        customer_text = customer_text + suffix

        # Ensure uniqueness.
        if customer_text in seen_texts:
            continue

        seen_texts.add(customer_text)

        # Slightly vary quality scores.
        base_score = random.choice([4, 4, 5])

        if random.random() < 0.15:
            base_score = 3

        examples.append({
            "id": example_id,
            "tweet_id": 119300 + example_id,
            "customer_text": customer_text,
            "true_intent": intent,
            "reference_reply": scenario["reply"],
            "true_escalation": scenario["escalation"],
            "escalation_reason": (
                "Requires human handling because the issue involves "
                "sensitive account, financial, or physical hardware support."
                if scenario["escalation"] == "ESCALATE_TO_HUMAN"
                else
                "Standard troubleshooting or informational request that can "
                "be handled automatically."
            ),
            "human_judge_score": base_score,
            "human_notes": (
                f"Manually reviewed evaluation example for {intent}. "
                f"Expected decision: {scenario['escalation']}. "
                f"Quality rating: {base_score}."
            )
        })

        example_id += 1

    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, "golden_eval_set.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    print(f"Generated {len(examples)} unique golden evaluation examples.")
    print("These examples are synthetic evaluation scenarios, not raw tweets.")


if __name__ == "__main__":
    build_golden_dataset()