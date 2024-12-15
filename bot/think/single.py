import random

from api.models import ChatMessage
from bot.think.flavors import THINKER_FLAVORS
from chat.chatter import SingleSentenceChatter
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)


class ThinkNode(SingleSentenceChatter):
    def __init__(self, flavor: str):
        super().__init__(
            name=flavor,
            system_message=" ".join([
                THINKER_FLAVORS[flavor],
                "Voice as inner dialogue.",  # Required because more than two voices in a two participant format
                "Don't use affirmatives."  # People don't tell themselves, "I agree, ...", they just think it
            ]))


class SingleThreadedMultiFlavorThinker:
    def __init__(self):
        self.nodes = [ThinkNode(flv) for flv in THINKER_FLAVORS.keys()]

    def add_first_message(self, text: str):
        for node in self.nodes:
            node.history.append(ChatMessage(role="user", content=text))

    def round_table(self):
        """
        Conduct a simulated inner voice "round-table". Unlike PairedThinkers that have back and forth discussions
        between two participants, with a SingleThreadedMultiFlavorThinker, an initial message is sent to all nodes
        and each participant node subsequently comment in turns. This is a more efficient way of incorporating a
        broad range of perspectives, where PairedThinkers might be better at exploding a single idea.

        :return:
        """

        # Order matters tremendously. Shuffling is needed. The first few nodes that comment have more influence
        # over the direction of the discussion. A static order applies the same biases every time.
        random.shuffle(self.nodes)

        # text        -> Flavor 1    Flavor 2    Flavor 3
        # response 1     Flavor 1 ->
        # response 2              <- Flavor 2 ->
        # response 3                          <- Flavor 3
        for idx, node in enumerate(self.nodes):
            reply = node.do_completion()
            logger.info(f"{node.name}: {reply}")
            for other_idx in [n for n in range(len(self.nodes)) if n != idx]:
                self.nodes[other_idx].history.append(ChatMessage(role="user", content=reply))

    def summarize(self):
        thoughts = []
        for flavored_node in self.nodes:
            thoughts.append({
                "thought": flavored_node.summarize(),
                "thinkers": [flavored_node.name]
            })
        return thoughts


if __name__ == "__main__":
    import argparse
    import json

    setup_logging(global_level="INFO")

    parser = argparse.ArgumentParser(description='Think about an idea.')
    parser.add_argument("-i", "--idea", help='Any idea, simple or complex.',
                        default="People are good.")
    args = parser.parse_args()

    thinker = SingleThreadedMultiFlavorThinker()
    thinker.add_first_message(args.idea)
    for i in range(3):
        thinker.round_table()
    print(json.dumps({"thoughts": thinker.summarize()}, indent=4))
    # print(json.dumps([m.model_dump() for m in random.choice(thinker.nodes).history], indent=4))
