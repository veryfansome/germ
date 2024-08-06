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
    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "on"}, messages=[
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
            content="""Make me a map that shows animal migration flows based on the following coordinates:

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
"""
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
            content="""Dealing with neighbors who leave their dog's waste on your property can be frustrating. Here are some steps you can take to address the situation:

1. **Document the Issue**: Take photos or videos of the dog waste on your property. This can serve as evidence if you need to escalate the issue.

2. **Communicate Directly**: Approach your neighbors in a friendly and non-confrontational manner. Let them know about the problem and ask them to clean up after their dog. Sometimes, people are unaware of the impact of their actions.

3. **Post Signs**: Consider putting up polite signs in your yard asking dog owners to clean up after their pets. This can serve as a gentle reminder.

4. **Provide Waste Bags**: Sometimes, providing a small station with waste bags can encourage dog owners to pick up after their pets.

5. **Contact Animal Control or Local Authorities**: If the problem persists, you can contact your local animal control or municipal authorities. Many places have ordinances requiring pet owners to clean up after their animals, and there may be fines for non-compliance.

6. **Install a Fence or Barrier**: If feasible, installing a fence or barrier can prevent dogs from entering your yard.

7. **Use Deterrents**: There are commercial products available that can deter dogs from entering your yard. These are usually safe and non-toxic.

8. **Seek Mediation**: If direct communication doesn't work, consider seeking mediation services. Many communities offer mediation to help resolve disputes between neighbors.

Remember to stay calm and respectful throughout the process. Maintaining a good relationship with your neighbors is important, and a friendly approach is often the most effective way to resolve the issue.
"""
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
            content="""I've written a children's book and need some ideas for cover art. It's a collection of poems about a character called, "Benny the Bear".

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
"""
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
