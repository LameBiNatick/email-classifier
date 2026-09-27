import random
import pandas as pd

# Expanded sample texts for each category
spam_samples = [
    "Win a FREE iPhone now!!! Limited time offer",
    "Congratulations! You've won a $1000 Walmart gift card",
    "Claim your lottery prize immediately!!!",
    "Earn money from home, no experience needed",
    "Act now to secure your cash reward",
    "You have been selected for a free vacation package",
    "Exclusive deal just for you, click this link",
    "Your account is compromised, verify immediately",
    "Get rich quick with this investment opportunity",
    "Congratulations, you are a lucky winner today"
]

promotions_samples = [
    "Huge discounts on electronics this weekend",
    "Special festive sale, buy 1 get 1 free",
    "Check out our new summer clothing collection",
    "Limited-time offer on premium subscriptions",
    "Mega sale on shoes and apparel",
    "Flash sale: Up to 70% off on select items",
    "Subscribe now and get 20% off your first order",
    "Valentine's Day special offers just for you",
    "Exclusive access to members-only discounts",
    "Year-end clearance sale starts today"
]

updates_samples = [
    "Your account password was updated successfully",
    "Your order has been shipped and will arrive soon",
    "System maintenance scheduled for tonight",
    "Your monthly statement is now available",
    "Meeting has been rescheduled to 3 PM tomorrow",
    "Your subscription will renew automatically",
    "Security alert: new login from unknown device",
    "Software update available for your device",
    "Delivery delayed due to weather conditions",
    "Your ticket has been confirmed"
]

personal_samples = [
    "Hey, are we still on for dinner tonight?",
    "Happy Birthday! Wishing you a great year ahead",
    "Let's catch up over coffee tomorrow",
    "Can you send me the notes from class?",
    "Good luck on your exams next week!",
    "Call me when you get home",
    "Thanks for helping me with my project",
    "Are you free to hang out this weekend?",
    "Don’t forget to bring your laptop tomorrow",
    "See you at the party tonight!"
]

# Function to expand with slight variations
def expand_samples(base_list, n):
    expanded = []
    for _ in range(n):
        text = random.choice(base_list)
        # add minor variation
        if random.random() > 0.5:
            text += random.choice(["!!!", " :) ", ".", " Please check."])
        expanded.append(text)
    return expanded

# Generate ~100 samples per class
spam = expand_samples(spam_samples, 100)
promotions = expand_samples(promotions_samples, 100)
updates = expand_samples(updates_samples, 100)
personal = expand_samples(personal_samples, 100)

# Combine into dataframe
data = {
    "text": spam + promotions + updates + personal,
    "category": ["spam"]*100 + ["promotions"]*100 + ["updates"]*100 + ["personal"]*100
}

df_expanded = pd.DataFrame(data)
csv_path_expanded = "emails_expanded.csv"   # save locally in the project folder
df_expanded.to_csv(csv_path_expanded, index=False)

print(f"✅ Expanded dataset saved as {csv_path_expanded}")

