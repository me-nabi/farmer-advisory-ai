"""
Dataset Generation Script for Farmer Crop Advisory AI
Generates Hindi agricultural Q&A pairs using GPT-4o-mini
Usage: python scripts/generate_data.py --api_key YOUR_OPENAI_KEY
"""
import openai, json, time, os, argparse

CROPS = [
    "गेहूं (wheat)", "धान (rice)", "मक्का (maize)", "कपास (cotton)",
    "गन्ना (sugarcane)", "सरसों (mustard)", "चना (chickpea)", "अरहर (pigeon pea)",
    "सोयाबीन (soybean)", "मूंगफली (groundnut)", "उड़द (black gram)", "मूंग (green gram)",
    "आलू (potato)", "टमाटर (tomato)", "प्याज (onion)", "मिर्च (chili)",
    "बैंगन (brinjal)", "भिंडी (okra)", "गोभी (cauliflower)", "पालक (spinach)",
    "आम (mango)", "केला (banana)", "अंगूर (grapes)", "संतरा (orange)",
    "अमरूद (guava)", "हल्दी (turmeric)", "अदरक (ginger)", "लहसुन (garlic)",
    "धनिया (coriander)", "जीरा (cumin)", "लौकी (bottle gourd)", "करेला (bitter gourd)",
    "तरबूज (watermelon)", "खीरा (cucumber)", "पपीता (papaya)", "अनार (pomegranate)",
]

SUBTOPICS = [
    "प्रमुख रोग — लक्षण, पहचान और रासायनिक उपचार (fungicide/bactericide names और dosage ml/liter)",
    "प्रमुख कीट — पहचान और कीटनाशक छिड़काव (insecticide names और dosage)",
    "बुवाई/रोपाई — सही समय, बीज दर (kg/hectare), बीज उपचार विधि",
    "उर्वरक प्रबंधन — NPK dosage (kg/hectare), DAP, Urea, MOP कब और कितना डालें",
    "सिंचाई प्रबंधन — कितनी सिंचाई, किस अवस्था में, कितना पानी",
    "खरपतवार नियंत्रण — weedicide names, dosage, छिड़काव का समय",
    "कटाई और भंडारण — सही समय, विधि, भंडारण में कीट नियंत्रण",
    "जैविक नियंत्रण — Trichoderma, NPV, neem oil, Beauveria bassiana का उपयोग",
]

GENERAL_TOPICS = [
    "मिट्टी परीक्षण — pH, EC, NPK कैसे जांचें, मिट्टी स्वास्थ्य कार्ड",
    "वर्मीकम्पोस्ट बनाने की पूरी विधि",
    "ड्रिप सिंचाई — स्थापना, लागत, सब्सिडी, रखरखाव",
    "फसल चक्र — धान-गेहूं, कपास-गेहूं, सोयाबीन-गेहूं rotation",
    "जैव उर्वरक — Rhizobium, Azotobacter, PSB का उपयोग",
    "सूक्ष्म पोषक तत्व — Zinc, Boron, Iron, Manganese की कमी के लक्षण",
    "कीटनाशक छिड़काव की सुरक्षित विधि — PPE, समय, wind direction",
    "प्रधानमंत्री फसल बीमा योजना (PMFBY) — पात्रता, प्रीमियम, दावा प्रक्रिया",
    "PM किसान सम्मान निधि — ₹6000/वर्ष, किश्त, eKYC, आवेदन",
    "किसान क्रेडिट कार्ड (KCC) — आवेदन, ब्याज दर, लिमिट",
    "e-NAM (राष्ट्रीय कृषि बाजार) — ऑनलाइन मंडी, रजिस्ट्रेशन",
    "DAP vs SSP — कौन सा उर्वरक बेहतर, कब कौन सा डालें",
    "रासायनिक खेती vs जैविक खेती — फायदे, नुकसान, लागत तुलना",
    "हाइब्रिड बीज vs देसी बीज — क्या अंतर है",
    "खरीफ सीजन की पूरी समय सारणी — बुवाई से कटाई तक",
    "रबी सीजन की पूरी समय सारणी — बुवाई से कटाई तक",
]


def generate_qa(client, topic, retries=3):
    prompt = f"""You are an expert Indian agricultural scientist.

Generate exactly 10 detailed question-answer pairs in HINDI about:
"{topic}"

CRITICAL RULES:
- Simple Hindi that uneducated farmers understand
- MUST contain specific: chemical names, exact dosage (ml/liter or kg/hectare), timing
- Factually accurate based on ICAR/KVK guidelines
- Every answer ends with: "अधिक जानकारी के लिए अपने नजदीकी KVK से संपर्क करें।"
- 3-5 sentences per answer, practical and actionable

Return ONLY JSON array:
[{{"instruction": "सवाल?", "input": "", "output": "जवाब।"}}]"""

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=2500
            )
            content = r.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(3)
    return []


def main(args):
    client = openai.OpenAI(api_key=args.api_key)

    # Build all topics
    all_topics = []
    for crop in CROPS:
        for sub in SUBTOPICS:
            all_topics.append(f"{crop} — {sub}")
    all_topics.extend(GENERAL_TOPICS)

    # Resume from existing progress
    save_file = args.output
    all_records = []
    start_from = 0

    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        start_from = len(all_records) // 10
        print(f"Resuming from topic {start_from + 1} ({len(all_records)} existing pairs)")

    print(f"Total topics: {len(all_topics)}")
    print(f"Remaining: {len(all_topics) - start_from}")
    print(f"Estimated cost: ~${(len(all_topics) - start_from) * 0.003:.2f}")

    for i, topic in enumerate(all_topics[start_from:], start_from + 1):
        pairs = generate_qa(client, topic)
        all_records.extend(pairs)
        if i % 10 == 0:
            print(f"[{i}/{len(all_topics)}] Total: {len(all_records)} pairs")
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Generated {len(all_records)} Q&A pairs → {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hindi agricultural Q&A dataset")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--output", default="data/generated_hindi_qa.json", help="Output file path")
    main(parser.parse_args())