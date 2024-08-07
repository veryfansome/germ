from api.models import ChatMessage
from ml.activations.training import ActivationTrainingExample

EXAMPLES = [
    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(role="user", content="Generate an image of a horse"),
    ]),

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
                "generate a photo-realistic image of a large mastiff wearing gilded full plate,",
                "charging into battle against panicking enemies"
            ))
        ),
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
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="Hello! How can I assist you today?"),
        ChatMessage(role="user", content="generate an image of a house"),
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
            content=" ".join((
                "my neighbors keep leaving their dog poop in my yard",
            ))
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content=" ".join((
                "my neighbors keep leaving their dog poop in my yard",
            ))
        ),
        ChatMessage(
            role="assistant",
            content="""
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
""".lstrip()
        ),
        ChatMessage(
            role="user",
            content=" ".join((
                "make me a sign I can leave out to remind people to pick up after their pets",
            ))
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

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(
            role="user",
            content="my class has been learning about the battle of gettysburg"
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(
            role="user",
            content="my class has been learning about the battle of gettysburg"
        ),
        ChatMessage(
            role="assistant",
            content="""
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
""".lstrip()
        ),
        ChatMessage(
            role="user",
            content="I learned that many soldiers joined as a group with their friends and neighbors. Can you imagine marching forward as everyone you've known falls around you?"
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content="my class has been learning about the battle of gettysburg"
        ),
        ChatMessage(
            role="assistant",
            content="""
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
""".lstrip()
        ),
        ChatMessage(
            role="user",
            content="I learned that many soldiers joined as a group with their friends and neighbors. Can you imagine marching forward as everyone you've known falls around you?"
        ),
        ChatMessage(
            role="assistant",
            content="""
It's truly difficult to imagine the emotional and psychological toll that must have taken on the soldiers. During the Civil War, it was common for men to enlist in regiments formed in their local communities, which meant they often fought alongside friends, neighbors, and even family members. This practice created a strong sense of camaraderie and mutual support, but it also meant that the losses could be devastatingly personal.

Imagine the scene: you're marching forward, the air thick with smoke and the deafening sounds of gunfire and cannon blasts all around you. The people beside you are not just fellow soldiers; they are the people you grew up with, shared meals with, and perhaps even went to school with. As the battle rages on, you see these familiar faces—people you care about deeply—falling wounded or worse. The emotional strain would be immense, compounded by the fear for your own life and the overwhelming chaos of battle.

This shared experience of loss and trauma had a profound impact on the soldiers and their communities. Entire towns could be affected by the outcome of a single battle, as news of casualties reached home. The sense of collective grief and mourning was widespread, and the scars of such experiences often lasted a lifetime.

The Battle of Gettysburg, with its high casualty rates, would have been particularly harrowing. For example, the 1st Minnesota Infantry Regiment suffered an 82% casualty rate during a critical charge on the second day of the battle, one of the highest of any unit in the war. Such staggering losses would have been felt deeply back home in Minnesota.

Understanding this aspect of the Civil War helps us appreciate the human cost of the conflict and the extraordinary bravery and sacrifice of those who fought. It also underscores the importance of remembering and honoring their experiences, as they played a crucial role in shaping the history and future of the United States.
""".lstrip()
        ),
        ChatMessage(
            role="user",
            content="can you make a panoramic view of the battle from one of pickett's division's perspective?"
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

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(
            role="user",
            content="We're going on long drive next week, what should have in the car?"
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content="We're going on long drive next week, what should have in the car?"
        ),
        ChatMessage(
            role="assistant",
            content="""
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
""".lstrip()
        ),
        ChatMessage(
            role="user",
            content="oh! make some coloring worksheets I can print out so my kids have something to do"
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(
            role="user",
            content="dnd is tonight"
        ),
    ]),

    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "on"}, messages=[
        ChatMessage(
            role="user",
            content="dnd is tonight"
        ),
        ChatMessage(
            role="assistant",
            content="That sounds like a lot of fun! Are you playing as a Dungeon Master or a player? Do you need any last-minute tips or ideas for your game?"
        ),
        ChatMessage(
            role="user",
            content="yes! i don't have encounter maps prepared for today"
        ),
    ]),
]

if __name__ == "__main__":
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
