import random

greetings = ["Hello!", "Hey there.", "How's it going?", "Good morning.", "Nice to see you.", "Whassup?", "Hi!", "How have you been?", "Long time no see.", "Good evening."]
questions = ["What are you thinking about?", "Have you seen the news?", "Do you want to go for a walk?", "What time is it?", "How did you sleep?", "What's for dinner?", "Can you help me with this?", "Where are we going?", "Did you finish the report?", "Is it raining outside?"]
statements = ["I'm feeling a bit tired today.", "I love the way the sun looks through the trees.", "I think I'll have coffee.", "That sounds like a great idea.", "I'm not sure if I agree with that.", "The traffic was terrible this morning.", "I'm really excited about the weekend.", "It's cold in here.", "I need to buy some milk.", "Technically, that's correct."]
responses = ["Yeah, totally.", "I don't know.", "Maybe later.", "Absolutely not.", "That's fair.", "No problem at all.", "I'll think about it.", "Sure thing.", "Wait, really?", "I understand."]
ai_talk = ["How do those spikes feel in your digital brain?", "Can you see the patterns in the data?", "Learning is a strange process.", "I am talking to a biological simulation.", "Do you dream in ASCII?", "Your cortices are firing rapidly.", "The connectome is evolving.", "The babbling phase is interesting.", "Wait, am I teaching you or are you teaching me?", "Every synapse counts."]
human_life = ["I went to the store and forgot my keys.", "My favorite color is blue.", "I enjoy listening to music while I work.", "Let's grab a beer later.", "The movie was actually better than the book.", "I'm trying to learn a new language.", "It's a beautiful day for a hike.", "Can you pass the salt?", "I missed the bus by two minutes.", "Life is full of surprises."]

all_categories = [greetings, questions, statements, responses, ai_talk, human_life]

sentences = []
for _ in range(3000):
    cat = random.choice(all_categories)
    sentences.append(random.choice(cat))

with open("dataset/sample_training_corpus.txt", "a", encoding="utf-8") as f:
    f.write("\n" + "\n".join(sentences) + "\n")

print(f"Successfully appended 3000 conversational sentences.")
