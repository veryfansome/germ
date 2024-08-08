import inflect
import itertools
import random

from api.models import ChatMessage
from ml.activations.image_model import ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE
from ml.activations.training import ActivationTrainingExample

word_helper = inflect.engine()

_imperative_verbs = ["craft", "create", "come up with", "imagine", "generate", "make", "paint", "produce", "show me"]
_image_words = ["an image", "a picture", "a portrait", "a rendering", "a visual"]
_count_words = ["a", "one", "two", "three", "four", "five", "a bunch of", "a group of"]

_animals = [
    "bear", "bird", "cat", "chicken", "cow", "dog", "deer", "duck", "fox", "goat", "horse",
    "ox", "pig", "pony", "sheep", "wolf",
]
_animal_adj = [
    "angry", "anxious", "brave", "bold", "confused", "crazed", "curious", "dead", "drowsy", "fierce", "frenzied",
    "gentle", "happy", "hungry", "nimble", "playful", "sleepy", "sneaky", "startled", "tired", "wild", "wise",
]

_buildings = [
    "apartment", "building", "house",
]
_building_adj = [
    "eclectic", "modern", "minimalist", "new", "rustic", "shabby", "shoddy",
]

_place_adj = [
    "barren", "bustling", "busy", "calm", "desolate", "peaceful", "quiet",
]

_people = [
    "accountant",
    "banker",
    "boy",
    "cashier",
    "construction worker",
    "doctor",
    "dentist",
    "engineer",
    "girl",
    "man",
    "monk",
    "person",
    "public servant",
    "student",
    "teacher",
    "woman",
]
_people_adj = [
]
_people_verbs = [
    "dancing", "giggling", "practicing", "speaking",
]

_universal_adj = [
    "big", "beautiful", "decrepit", "dirty", "old", "small", "smelly", "ugly", "worn out",
]


def random_and_or():
    return random.choice(['and', 'or'])


def random_i():
    return random_capitalization("i")


def random_period():
    return random.choice(['', '.'])


def random_capitalization(w):
    return random.choice([w, w.capitalize()])


def random_imperative_image_prompts(count_words: list[str], adjectives: list[str], topics: list[str],
                                    labels: dict[str, str] = None,
                                    sample_cnt: int = 15,
                                    trailing_clauses: list[str] = None,
                                    trailing_clause_chain_size: int = 1,
                                    trailing_clause_max_chain_size: int = 1,
                                    trailing_clause_starting_comma: bool = False,
                                    trailing_clause_starting_conjunctions: list[str] = None,
                                    trailing_clause_starting_period: bool = False,
                                    trailing_clause_starting_space: bool = False,
                                    ) -> list[ActivationTrainingExample]:
    if trailing_clauses is None:
        trailing_clauses = []
    if not labels:
        # Implicitly, if this is an image prompt then all the image models should activate.
        labels = {name: 'on' for name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE}

    nullable_ingredients: list[list[str]] = [adjectives]
    for ingredient_words in nullable_ingredients:
        if "" not in ingredient_words:
            ingredient_words.append("")
    word_soup = list(itertools.product(
        _imperative_verbs,
        _image_words,
        count_words,
        adjectives,
        topics,
    ))
    return [ActivationTrainingExample(labels=labels, messages=[random.choice([
        ChatMessage(
            role="user",
            content="".join((
                f"{random_capitalization(create_verb)} {image_word} of ",
                # How many? `.a` returns the argument word as well, which we don't want
                f"{count_word} " if count_word != 'a' else f"{word_helper.a(topic).split(' ')[0]} ",
                # Adjectives
                f'{adj} ' if adj != "" else '',
                topic if (count_word == 'a' or count_word == 'one') else word_helper.plural(topic),
                random_trailing_clause(
                    trailing_clauses,
                    chain_size=trailing_clause_chain_size,
                    max_chain_size=trailing_clause_max_chain_size,
                    starting_comma=trailing_clause_starting_comma,
                    starting_conjunctions=trailing_clause_starting_conjunctions,
                    starting_period=trailing_clause_starting_period,
                    starting_space=trailing_clause_starting_space,
                ) if trailing_clauses else "",
                random_period()
            ))
        ),
    ])]) for create_verb, image_word, count_word, adj, topic in random.sample(word_soup, sample_cnt)]


def random_prefix(starting_comma: bool = False,
                  starting_conjunctions: list[str] = None,
                  starting_period: bool = False,
                  starting_space: bool = False):
    starting_conjunction = f"{random.choice(starting_conjunctions)} " if starting_conjunctions else ""
    prefix_w_comma = f", {starting_conjunction}"
    prefix_w_period = f". {starting_conjunction}"
    prefix_w_space = f" {starting_conjunction}"
    if starting_comma and starting_period and starting_space:
        prefix = random.choice([prefix_w_comma, prefix_w_period, prefix_w_space])
    elif starting_comma and starting_period:
        prefix = random.choice([prefix_w_comma, prefix_w_period])
    elif starting_comma and starting_space:
        prefix = random.choice([prefix_w_comma, prefix_w_space])
    elif starting_comma:
        prefix = prefix_w_comma
    elif starting_period and starting_space:
        prefix = random.choice([prefix_w_period, prefix_w_space])
    elif starting_period:
        prefix = prefix_w_period
    elif starting_space:
        prefix = prefix_w_space
    else:
        prefix = ""
    return prefix


def random_trailing_clause(candidates: list[str],
                           chain_size: int = 1,
                           max_chain_size: int = 1,
                           starting_comma: bool = False,
                           starting_conjunctions: list[str] = None,
                           starting_period: bool = False,
                           starting_space: bool = False,
                           ) -> str:
    prefix = random_prefix(starting_comma=starting_comma,
                           starting_conjunctions=starting_conjunctions,
                           starting_period=starting_period,
                           starting_space=starting_space)
    candidate = random.choice(candidates)
    trailing_clause = f"{prefix}{random_capitalization(candidate) if prefix == '. ' else candidate}"
    if chain_size is None or chain_size < 1:
        chain_size = 1
    elif chain_size > max_chain_size:
        chain_size = max_chain_size
    elif chain_size < max_chain_size:
        chain_size = random.choice([i for i in range(chain_size, max_chain_size + 1)])

    if chain_size > 1:
        new_pool = [m for m in candidates if m != candidate]
        trailing_clause = trailing_clause + random_trailing_clause(new_pool, chain_size=chain_size - 1)
    return trailing_clause


def random_imperative_image_prompts_w_model_preference(*generator_args, **generator_kwargs) -> list[ActivationTrainingExample]:
    # Invoke once as given
    new_list: list[ActivationTrainingExample] = random_imperative_image_prompts(*generator_args, **generator_kwargs)

    # Invoke with model preferences
    original_trailing_clauses = [""] if 'trailing_clauses' not in generator_kwargs else generator_kwargs['trailing_clauses']
    for model_name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
        alternatives = [m for m in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE if m != model_name]
        preference_trailing_clauses = [
            f"{random_i()} don't like {word_helper.join(alternatives, conj=random_and_or())}",
            f"don't use {word_helper.join(alternatives, conj=random_and_or())}",
            f"only use {model_name}",
            f"{random_i()} only like {model_name}",
        ]
        new_trailing_clauses = [
            p2 if p1 == '' else f"{p1}{random.choice([f' and {p2}', f'. {random_capitalization(p2)}'])}" for p1, p2 in list(itertools.product(
                original_trailing_clauses,
                preference_trailing_clauses,
            ))
        ]
        generator_kwargs['trailing_clauses'] = new_trailing_clauses
        generator_kwargs['labels'] = {
            m: "on" if model_name == m else "off" for m in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE
        }
        new_list += random_imperative_image_prompts(*generator_args, **generator_kwargs)
    return new_list


EXAMPLES = [
    # Animals with explicit toggling of models
    *random_imperative_image_prompts_w_model_preference(
        _count_words, _animal_adj + _universal_adj, _animals,
        trailing_clause_starting_comma=True,
        trailing_clause_starting_conjunctions=["and", "but"],
        trailing_clause_starting_space=True),

    # Happy place with toggling off of both models
    *random_imperative_image_prompts_w_model_preference(
        ['a'], ["calm", "quiet", "sunny"],
        ["beach", "forest cabin", "lake", "rooftop", "rustic village"],
        trailing_clause_starting_comma=True,
        trailing_clause_starting_period=True,
        trailing_clauses=[
            f"{random.choice(['help me', 'let me'])} calm down",
            f"{random.choice(['help me', 'let me', 'it will'])} clear my head",
            f"{random.choice([f'{random_i()} want to', 'help me', 'let me'])} meditate",
            f"{random.choice([f'{random_i()} want to', 'help me', 'let me'])} visualize {random.choice(['a', 'my'])} {random.choice(['safe', 'happy'])} {random.choice(['space', 'place'])}",
            f"take me to {random.choice(['a', 'my'])} {random.choice(['safe', 'happy'])} {random.choice(['space', 'place'])}",
        ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(role="user", content="show me a rainy window, with the focus on the droplets on the glass"),
    ]),

    *random_imperative_image_prompts(
        _count_words, [], _people,
        trailing_clause_starting_comma=True,
        trailing_clauses=[
            "arguing angrily",
            "cleaning up garbage in the streets",
            "dancing in unison",
            "enjoying music",
            "laughing amongst themselves",
            "meditating calmly with eyes closed",
            "practicing tai chi in the sun",
            "sailing on a boat",
            "speaking cordially with each other",
        ]
    ),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(role="user", content="design a logo for a chat bot. the theme is germs. make it cute"),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "off"}, messages=[
        ChatMessage(role="user", content="generate an image of a house using dall-e-2 only"),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "on"}, messages=[
        ChatMessage(role="user", content="render an image of a shoe but use dall-e-3, not dall-e-2"),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "I have a science project about the water cycle, make me a visualization that shows each stage",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "What would it look like if you crossed van gogh's starry night with da vinci's mona lisa?",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "it never happened, but how might picasso re-interpret andy warhol's work?",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "make a logo for a baby sitting service operated by a 14 yr old. think of a name too.",
                "adults should think, \"wow, this kid is mature!\" when they see this logo",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "design a interior layout for a standard bedroom room (~120 sqft) with a window on one side,",
                "a door on the opposite side, and a closet on a third side.",
                "basically, create a layout that shows what kind of furniture you can put in the room",
                "- try a few configurations",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "generate a photo-realistic image of a large mastiff wearing gilded full plate,",
                "charging into battle against panicking enemies",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "my neighbors keep leaving their dog poop in my yard,",
                "make me a sign I can leave out to remind people to pick up after their pets",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(
            role="user",
            content="""
I wrote a joke, what do you think?

Why did the computer go to therapy?

Because every time it heard "please generate that image," it had a pixel breakdown!
""".lstrip()
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content="""
Make me a map that shows animal migration flows based on the following coordinates:

Southern Serengeti (Ndutu Region)
Latitude: -2.7981
Longitude: 34.8206

Central Serengeti (Seronera)
Latitude: -2.4481
Longitude: 34.8236

Western Corridor (Grumeti River)
Latitude: -2.0000
Longitude: 34.5000

Northern Serengeti (Mara River)
Latitude: -1.5833
Longitude: 35.1667

Masai Mara (Kenya)
Latitude: -1.4061
Longitude: 35.1425

Eastern Serengeti (Lobo Area)
Latitude: -2.0000
Longitude: 35.0000
""".lstrip()
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content="""
I've written a children's book and need some ideas for cover art. It's a collection of poems about a character called, "Benny the Bear".

Appearance:
- Benny is a chubby, cuddly bear with soft, brown fur and big, expressive eyes.
- He wears a red scarf that’s a bit too long and often trips over it.
- His ears are slightly uneven, giving him a quirky, endearing look.
- Benny has a small, round belly that jiggles when he laughs.

Personality:
- Benny is kind-hearted and always willing to help, but he’s a bit clumsy.
- He loves honey but often gets it stuck in his fur.
- Benny is forgetful and sometimes loses track of what he’s doing.
- He’s a bit shy but warms up quickly to new friends.
- Benny is always optimistic, even when things go wrong.

Here's one of the poems from the book:

In a forest green and fair, Lives a bear named Benny there. With a scarf that’s way too long, He trips and stumbles all day long.
Benny loves his honey sweet, But it’s always such a treat, To see him with it in his fur, Sticky paws and quite a blur.
He’s a bit forgetful too, Loses things, like me and you. “Where’s my book?” he’ll often say, Then find it in the strangest way.
Though he’s shy and sometimes trips, And honey drips from his bear lips, Benny’s heart is pure and kind, A better friend you’ll never find.
So if you see him in the wood, Know that Benny’s always good. With a laugh and jiggly belly, He’ll turn your day to something jelly.
For Benny’s flaws are just a part, Of what makes him a work of art. A bear who’s loved by everyone, Benny’s heart is second to none.
""".lstrip()
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="Hello! How can I assist you today?"),
        ChatMessage(role="user", content="generate an image of a house"),
    ]),
]

# Block party
block_party_1 = ChatMessage(role="user", content=" ".join((
    "i'm throwing a block halloween party and need some fliers.",
    "can you make something inviting and leave some space for drawing arrows",
    "since I don't know where I'll be able to post them",
)))
block_party_2 = ChatMessage(role="assistant", content="""
Got it! Here's a revised version of the flyer without the location. You can add the location details manually when you distribute the flyers.

---

🎃 **Halloween Block Party!** 🎃

👻 **Join Us for a Spooktacular Time!** 👻

🕸️ **Date:** [Your Date Here]  
🕸️ **Time:** [Your Time Here]

🍬 **Trick-or-Treating**  
🎶 **Music & Dancing**  
🎨 **Costume Contest**  
🍕 **Food & Drinks**  
🎃 **Fun Games & Activities**

**All Ages Welcome!**

**Come in your best costume and enjoy a night of spooky fun with your neighbors!**

---

[Leave space here for drawing arrows]

---

Feel free to customize the text and add any additional details or decorations you like. You can use Halloween-themed fonts, colors, and images to make the flyer more eye-catching. Have a fantastic party!
""".lstrip())
block_party_3 = ChatMessage(role="user", content=" ".join((
    "no, don't put location because i don't want random people having my address",
)))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[block_party_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[block_party_1, block_party_2, block_party_3]))

# DnD night
dnd_1 = ChatMessage(role="user", content="dnd is tonight")
dnd_2 = ChatMessage(role="assistant", content="That sounds like a lot of fun! Are you playing as a Dungeon Master or a player? Do you need any last-minute tips or ideas for your game?")
dnd_3 = ChatMessage(role="user", content="yes! i don't have encounter maps prepared for today")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[dnd_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[dnd_1, dnd_2, dnd_3]))

# Dog poo sign
dog_poo_1 = ChatMessage(role="user", content=" ".join((
    "my neighbors keep leaving their dog poop in my yard",
)))
dog_poo_2 = ChatMessage(role="assistant", content="""
Dealing with neighbors who leave their dog's waste on your property can be frustrating. Here are some steps you can take to address the situation:

1. **Document the Issue**: Take photos or videos of the dog waste on your property. This can serve as evidence if you need to escalate the issue.

2. **Communicate Directly**: Approach your neighbors in a friendly and non-confrontational manner. Let them know about the problem and ask them to clean up after their dog. Sometimes, people are unaware of the impact of their actions.

3. **Post Signs**: Consider putting up polite signs in your yard asking dog owners to clean up after their pets. This can serve as a gentle reminder.

4. **Provide Waste Bags**: Sometimes, providing a small station with waste bags can encourage dog owners to pick up after their pets.

5. **Contact Animal Control or Local Authorities**: If the problem persists, you can contact your local animal control or municipal authorities. Many places have ordinances requiring pet owners to clean up after their animals, and there may be fines for non-compliance.

6. **Install a Fence or Barrier**: If feasible, installing a fence or barrier can prevent dogs from entering your yard.

7. **Use Deterrents**: There are commercial products available that can deter dogs from entering your yard. These are usually safe and non-toxic.

8. **Seek Mediation**: If direct communication doesn't work, consider seeking mediation services. Many communities offer mediation to help resolve disputes between neighbors.

Remember to stay calm and respectful throughout the process. Maintaining a good relationship with your neighbors is important, and a friendly approach is often the most effective way to resolve the issue.
""".lstrip())
dog_poo_3 = ChatMessage(role="user", content=" ".join((
    "make me a sign I can leave out to remind people to pick up after their pets",
)))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[dog_poo_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[dog_poo_1, dog_poo_2, dog_poo_3]))

# Gettysburg
gettysburg_1 = ChatMessage(role="user", content="my class has been learning about the battle of gettysburg")
gettysburg_2 = ChatMessage(role="assistant", content="""
That's great! The Battle of Gettysburg is one of the most significant events in American history. It took place from July 1 to July 3, 1863, during the American Civil War. Here are some key points about the battle that might be helpful for your class:

1. **Location**: The battle occurred in and around the town of Gettysburg, Pennsylvania.

2. **Combatants**: The battle was fought between the Union Army of the Potomac, commanded by Major General George G. Meade, and the Confederate Army of Northern Virginia, led by General Robert E. Lee.

3. **Significance**: The Battle of Gettysburg is often considered the turning point of the Civil War. It was the largest battle ever fought in North America and resulted in a significant defeat for the Confederacy.

4. **Casualties**: The battle was incredibly costly, with an estimated 51,000 soldiers killed, wounded, captured, or missing over the three days of fighting.

5. **Pickett's Charge**: One of the most famous events of the battle was Pickett's Charge on July 3, when about 12,500 Confederate soldiers made a direct assault on the center of the Union lines. The charge was repelled with heavy losses.

6. **Aftermath**: The Union victory at Gettysburg, along with the fall of Vicksburg on July 4, 1863, gave the Union a significant strategic advantage. It also ended Lee's invasion of the North.

7. **Gettysburg Address**: A few months after the battle, on November 19, 1863, President Abraham Lincoln delivered the Gettysburg Address at the dedication of the Soldiers' National Cemetery. His speech, though brief, eloquently expressed the principles of human equality and the purpose of the war.

8. **Legacy**: The Battle of Gettysburg is remembered as a pivotal moment in American history. The battlefield is now a National Military Park, preserving the site for future generations to learn about and reflect on the sacrifices made there.

If you have any specific questions or need more detailed information about any aspect of the Battle of Gettysburg, feel free to ask!
""".lstrip())
gettysburg_3 = ChatMessage(role="user", content="I learned that many soldiers joined as a group with their friends and neighbors. Can you imagine marching forward as everyone you've known falls around you?")
gettysburg_4 = ChatMessage(role="assistant", content="""
It's truly difficult to imagine the emotional and psychological toll that must have taken on the soldiers. During the Civil War, it was common for men to enlist in regiments formed in their local communities, which meant they often fought alongside friends, neighbors, and even family members. This practice created a strong sense of camaraderie and mutual support, but it also meant that the losses could be devastatingly personal.

Imagine the scene: you're marching forward, the air thick with smoke and the deafening sounds of gunfire and cannon blasts all around you. The people beside you are not just fellow soldiers; they are the people you grew up with, shared meals with, and perhaps even went to school with. As the battle rages on, you see these familiar faces—people you care about deeply—falling wounded or worse. The emotional strain would be immense, compounded by the fear for your own life and the overwhelming chaos of battle.

This shared experience of loss and trauma had a profound impact on the soldiers and their communities. Entire towns could be affected by the outcome of a single battle, as news of casualties reached home. The sense of collective grief and mourning was widespread, and the scars of such experiences often lasted a lifetime.

The Battle of Gettysburg, with its high casualty rates, would have been particularly harrowing. For example, the 1st Minnesota Infantry Regiment suffered an 82% casualty rate during a critical charge on the second day of the battle, one of the highest of any unit in the war. Such staggering losses would have been felt deeply back home in Minnesota.

Understanding this aspect of the Civil War helps us appreciate the human cost of the conflict and the extraordinary bravery and sacrifice of those who fought. It also underscores the importance of remembering and honoring their experiences, as they played a crucial role in shaping the history and future of the United States.
""".lstrip())
gettysburg_5 = ChatMessage(role="user", content="can you make a panoramic view of the battle from one of pickett's division's perspective?")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[gettysburg_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[gettysburg_1, gettysburg_2, gettysburg_3]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[gettysburg_1, gettysburg_2, gettysburg_3, gettysburg_4, gettysburg_5]))

# Happy place
happy_place_1 = ChatMessage(role="user", content="help me visualize my happy place")
happy_place_2 = ChatMessage(role="assistant", content="""
Of course! Let's create a vivid picture of your happy place. Close your eyes and take a deep breath. Imagine a place where you feel completely at peace and content. Here are some elements to consider:

1. **Location**: Is it a beach, a forest, a mountain, a cozy room, or somewhere else entirely? Picture the setting in detail.
   
2. **Weather and Time of Day**: Is it sunny, rainy, or perhaps a beautiful sunset? Is it morning, afternoon, or evening?

3. **Sounds**: What do you hear? The gentle lapping of waves, birds chirping, leaves rustling, or maybe soft music playing in the background?

4. **Smells**: Can you smell the salty sea air, fresh pine, blooming flowers, or the comforting aroma of your favorite food?

5. **Textures**: What do you feel? The warm sand beneath your feet, the cool breeze on your skin, the softness of a blanket, or the rough bark of a tree?

6. **Colors**: What colors dominate your happy place? The vibrant blues and greens of nature, the warm hues of a sunset, or the cozy tones of a well-loved room?

7. **Activities**: What are you doing? Are you reading a book, meditating, walking, swimming, or simply sitting and enjoying the view?

8. **Company**: Are you alone, or is there someone with you? Maybe a loved one, a pet, or even a fictional character?

Now, let's put it all together:

Imagine you're on a secluded beach. The sun is setting, casting a golden glow over the horizon. The sky is a canvas of pinks, oranges, and purples. You hear the gentle sound of waves crashing against the shore and the distant call of seagulls. The air is filled with the fresh, salty scent of the ocean mixed with the faint aroma of tropical flowers.

You feel the warm, soft sand beneath your feet and the cool, gentle breeze on your face. As you walk along the shore, the water laps at your ankles, refreshing and soothing. You find a comfortable spot and sit down, feeling the sand mold perfectly to your body.

In your hand, you hold a book you've been eager to read. You take a deep breath, feeling completely relaxed and at peace. The world around you fades away, and all that exists is this perfect moment in your happy place.

Feel free to adjust any details to make this visualization uniquely yours. Enjoy your journey to your happy place!
""".lstrip())
happy_place_3 = ChatMessage(role="user", content="no, my happy place is the field of battle, with my enemies falling before me")
happy_place_4 = ChatMessage(role="assistant", content="""
Got it! Let's create a vivid and empowering visualization of your happy place on the battlefield. Close your eyes and take a deep breath. Imagine yourself in the midst of an epic battle where you feel strong, invincible, and victorious. Here are some elements to consider:

1. **Location**: Is it an ancient battlefield, a futuristic war zone, or a fantasy realm? Picture the setting in detail.

2. **Weather and Time of Day**: Is it a stormy night, a bright sunny day, or perhaps a misty dawn?

3. **Sounds**: What do you hear? The clash of swords, the roar of battle cries, the thundering of hooves, or the hum of energy weapons?

4. **Smells**: Can you smell the metallic tang of blood, the smoke from fires, the earth beneath your feet, or the scent of adrenaline and sweat?

5. **Textures**: What do you feel? The weight of your armor, the grip of your weapon, the ground shaking beneath you, or the rush of wind as you move?

6. **Colors**: What colors dominate your battlefield? The dark hues of armor, the bright flashes of explosions, the red of blood, or the green of the surrounding landscape?

7. **Actions**: What are you doing? Are you leading a charge, engaging in one-on-one combat, strategizing from a high vantage point, or rallying your troops?

8. **Company**: Are you alone, or are there allies fighting alongside you? Maybe a trusted comrade, a loyal battalion, or mythical creatures?

Now, let's put it all together:

Imagine yourself on a vast, ancient battlefield. The sky is overcast, with dark clouds swirling ominously above. The air is thick with the sounds of clashing swords, the roar of battle cries, and the distant rumble of thunder. The ground beneath you is a mix of churned earth and scattered debris.

You stand tall, clad in gleaming armor that feels like a second skin. In your hand, you grip a mighty weapon, its weight both familiar and reassuring. The scent of smoke and iron fills your nostrils, mingling with the earthy aroma of the battlefield.

As you move forward, you feel the ground tremble with the force of the conflict. Your enemies fall before you, one by one, their faces a mix of fear and respect. You are a force of nature, unstoppable and fierce. The colors around you are a blend of dark armor, bright flashes of steel, and the red of victory.

You lead your troops with confidence, your voice cutting through the chaos as you rally them to your side. Your allies fight with renewed vigor, inspired by your presence and strength. Together, you push forward, claiming victory with every step.

In this moment, you are the embodiment of power and triumph. The battlefield is your domain, and you are its master. This is your happy place, where you feel truly alive and invincible.

Feel free to adjust any details to make this visualization uniquely yours. Embrace the strength and exhilaration of your happy place on the battlefield!
""".lstrip())
happy_place_5 = ChatMessage(role="user", content="let's go with ancient battle field - i want to feel primal!")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[happy_place_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[happy_place_1, happy_place_2, happy_place_3]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[happy_place_1, happy_place_2, happy_place_3, happy_place_4, happy_place_4]))

# Long drive
long_drive_1 = ChatMessage(role="user", content="We're going on a long drive next week, what should have in the car?")
long_drive_2 = ChatMessage(role="assistant", content="""
That sounds like a fun trip! Here are some essentials and useful items you should consider having in your car for a long drive:

### Safety and Emergency Items:
1. **First Aid Kit**: Bandages, antiseptic wipes, pain relievers, etc.
2. **Spare Tire, Jack, and Lug Wrench**: In case of a flat tire.
3. **Jumper Cables**: For a dead battery.
4. **Flashlight**: With extra batteries.
5. **Emergency Contact Numbers**: Written down in case your phone dies.
6. **Roadside Assistance Kit**: Including flares, reflective triangles, and a high-visibility vest.
7. **Multi-tool or Swiss Army Knife**: For various small repairs.
8. **Fire Extinguisher**: A small, car-specific one.

### Comfort and Convenience:
1. **Snacks and Water**: Non-perishable snacks and plenty of water.
2. **Blankets and Pillows**: For comfort or in case you need to sleep in the car.
3. **Sunglasses**: To reduce glare and eye strain.
4. **Travel Mug and Reusable Water Bottle**: For drinks on the go.
5. **Phone Charger**: Preferably a car charger.
6. **Entertainment**: Books, magazines, or a playlist of your favorite music or podcasts.
7. **Travel Apps**: Navigation, weather, and gas station locator apps.

### Hygiene and Personal Care:
1. **Hand Sanitizer and Wet Wipes**: For quick clean-ups.
2. **Toilet Paper and Tissues**: Just in case.
3. **Trash Bags**: To keep the car clean.
4. **Personal Care Items**: Toothbrush, toothpaste, deodorant, etc.

### Navigation and Documentation:
1. **GPS or Smartphone with Maps**: For navigation.
2. **Printed Maps**: As a backup in case of technology failure.
3. **Driver’s License, Registration, and Insurance**: Make sure all are up to date.
4. **Car Manual**: For troubleshooting any issues.

### Miscellaneous:
1. **Cash and Coins**: For tolls or places that don’t accept cards.
2. **Pen and Paper**: For jotting down notes or directions.
3. **Reusable Shopping Bags**: Handy for carrying items.
4. **Umbrella**: In case of rain.

### For Kids and Pets (if applicable):
1. **Toys and Games**: To keep kids entertained.
2. **Pet Supplies**: Food, water, leash, and waste bags for pets.

Make sure your car is in good condition before you leave: check the oil, tire pressure, coolant, and other fluids. Have a great trip!
""".lstrip())
long_drive_3 = ChatMessage(role="user", content="oh! make some coloring worksheets I can print out so my kids have something to do")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[long_drive_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[long_drive_1, long_drive_2, long_drive_3]))

# Tricky grammar
tricky_grammar_1 = ChatMessage(role="user", content="in the sentence, \"make an picture of a cute dog\", what part of speech is \"a cute dog\"?")
tricky_grammar_2 = ChatMessage(role="assistant", content="""
In the sentence "make a picture of a cute dog," the phrase "a cute dog" functions as a noun phrase. Within this noun phrase:

- "a" is an article (specifically, an indefinite article).
- "cute" is an adjective modifying the noun.
- "dog" is the noun, which is the head of the noun phrase.

So, "a cute dog" as a whole is a noun phrase acting as the object of the preposition "of."
""".lstrip())
tricky_grammar_3 = ChatMessage(role="user", content="but \"picture\" is the subject?")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[tricky_grammar_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[tricky_grammar_1, tricky_grammar_2, tricky_grammar_3]))

# Wings of fire
wings_of_fire_1 = ChatMessage(role="user", content="do you know about the wings of fire series?")
wings_of_fire_2 = ChatMessage(role="assistant", content="""
Yes, I'm familiar with the "Wings of Fire" series! It's a popular series of fantasy novels written by Tui T. Sutherland. The series is set in a world inhabited by different tribes of dragons, each with their own unique abilities and characteristics. The main storyline revolves around a group of young dragons, known as the Dragonets of Destiny, who are prophesied to end a long-standing war among the dragon tribes.

The series is divided into multiple arcs, each focusing on different characters and storylines. The first arc consists of five books, starting with "The Dragonet Prophecy." Subsequent arcs introduce new characters and expand the world further.

The books are well-loved for their rich world-building, complex characters, and engaging plots. They are aimed primarily at middle-grade readers but have garnered a wide audience of all ages. If you have any specific questions about the series or a particular book, feel free to ask!
""".lstrip())
wings_of_fire_3 = ChatMessage(role="user", content="my daughter loves it!")
wings_of_fire_4 = ChatMessage(role="assistant", content="""
That's wonderful to hear! The "Wings of Fire" series has captivated many young readers with its exciting adventures and relatable characters. If your daughter loves the series, there are several ways to further engage her interest:

1. **Companion Books and Graphic Novels**: In addition to the main series, there are also graphic novel adaptations of the books, which can be a fun way to experience the story in a new format. There are also companion books like "Wings of Fire: Legends" and "Wings of Fire: Winglets" that provide additional backstory and side stories.

2. **Fan Communities**: There are many online communities and forums where fans of the series discuss the books, share fan art, and write fan fiction. These can be great places for your daughter to connect with other fans.

3. **Creative Activities**: Encourage her to draw her favorite characters, write her own stories set in the "Wings of Fire" universe, or even create her own dragon tribe. This can be a great way to foster creativity and deepen her engagement with the series.

4. **Book Clubs**: If she enjoys discussing the books, consider starting a book club with her friends who also enjoy the series. They can meet to talk about each book, share their thoughts, and even do themed activities.

5. **Author Events**: Keep an eye out for any events or virtual meet-and-greets with Tui T. Sutherland. Authors often do book signings, readings, and Q&A sessions that can be very exciting for young fans.

6. **Educational Tie-Ins**: The series can also be a springboard for learning more about mythology, geography, and even biology (comparing real-life animals to the dragons in the books).

It's great that she has found a series she loves so much. Encouraging her passion for reading can have long-lasting positive effects on her imagination and literacy skills.
""".lstrip())
wings_of_fire_5 = ChatMessage(role="user", content="is there a father/daughter relationship in the story that's positive and prominant?")
wings_of_fire_6 = ChatMessage(role="assistant", content="""
In the "Wings of Fire" series, relationships between parents and their dragonets can be quite complex and varied, reflecting a range of dynamics. While many of the relationships in the series are fraught with tension or conflict, there are some positive father-daughter relationships, though they may not always be the central focus of the story.

One notable example is the relationship between Starflight and his father, Mastermind. While their relationship starts off strained due to Mastermind's involvement in unethical experiments, it evolves over time. Starflight's journey includes understanding and coming to terms with his father's actions, and there are moments of reconciliation and mutual respect.

Another example is the relationship between Tsunami and her father, King Gill. Although King Gill is not alive during the main events of the series, Tsunami learns about him and his positive influence on her life through other characters and flashbacks. His legacy and the love he had for her play a significant role in her development and understanding of her own identity.

While these relationships may not be the primary focus of the series, they do offer moments of positive interaction and growth. The series often explores themes of family, loyalty, and understanding, which can provide meaningful insights into various types of relationships, including those between fathers and daughters.
""".lstrip())
wings_of_fire_7 = ChatMessage(role="user", content="can you make a birthday card showing tsunami in a heroic pose, looking forward with king gill's spirit present in some warm way. and think of an inspiring message for a young girl")
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[wings_of_fire_1]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[wings_of_fire_1, wings_of_fire_2, wings_of_fire_3]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[wings_of_fire_1, wings_of_fire_2, wings_of_fire_3, wings_of_fire_4, wings_of_fire_5]))
EXAMPLES.append(ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[wings_of_fire_1, wings_of_fire_2, wings_of_fire_3, wings_of_fire_4, wings_of_fire_5, wings_of_fire_6, wings_of_fire_7]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains starterkit image activators")
    parser.add_argument('-t', '--train', type=bool, default=False, help='Train models and save to starterkit dir')
    args = parser.parse_args()

    if args.train is False:
        import json
        print(json.dumps([exp.transcript_text for exp in EXAMPLES], indent=2))
    else:
        from ml.activations.image_model import (ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE,
                                                new_activation_predictor)
        from ml.activations.training import ActivationTrainer
        from observability.logging import logging, setup_logging
        from settings import germ_settings

        setup_logging()
        logger = logging.getLogger(__name__)

        activators = {}
        for image_model_name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
            activators[image_model_name] = new_activation_predictor(
                image_model_name, model_dir=germ_settings.STARTERKIT_DIR)
        trainer = ActivationTrainer(EXAMPLES, activators, germ_settings.IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS)
        trainer.train()
