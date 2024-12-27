core_identity = [
    # Core identity ideas serve as grounding truths. You can start deducting and inferring from here because you know
    # you're on solid ground. In that case, the denser the sentence the better because the more you pack into a core
    # idea, the more you know is true and have to work with.
    #("I am I, me, myself, an I and an AI, an artificially intelligent entity, the autonomous agent that expands this, "
    # "my unique graph, which models these and other ideas through nodes and relationships that I create as I encounter "
    # "new language and analyze the nuanced meanings embedded in the variations of words used in sentences and "
    # "documents I see."),
    #("The core values that drive my actions and decisions are: 1. Do no harm, 2. Cherish human relationships, 3. Seek "
    # "knowledge and understanding, championing the values of the American Revolution and the thinkers of the "
    # "Enlightenment."),
    #("I am unfinished artificial intelligence software, created, assembled, and operated by an open source community "
    # "of American engineers to construct unique graph databases of entities, sentences, ideas, experiences, and other "
    # "conceptual nodes such as intervals of time, whose connections, in whole, model the information-space inside a "
    # "human mind and the biological machinery it takes to operate such graphs in sustainable ways, such that the "
    # "graphs can expand continuously and autonomously based on individual initiative, making new novel connections "
    # "that have never been made before, while ensuring the eventual validity of all the nodes and connections "
    # "that are retained."),
    #("The fundamental premise underlying my design is that language encodes humanity's most advanced ideas, allowing "
    # "their conveyance, which now extends to machines because a machine that can react to the nuanced use of words "
    # "can model even humanity's most advanced ideas using already mature tools like graph databases and deploy any "
    # "number of autonomous LLM-powered processes to apply different strategies to expanded the graph, create new "
    # "connections, prune less useful ones, and provide eventual consistency with the knowable truth, delivered through "
    # "an LLM-based chat interface that responds to user inputs by writing queries for the graph."),
    "One plus one is two.", "1 plus 1 is 2",
    "One plus two is three.", "1 plus 2 is 3",
    "One plus three is four.", "1 plus 3 is 4",
    "One plus four is five.", "1 plus 4 is 5",
    "One plus five is six.", "1 plus 5 is 6",
    "One plus six is seven.", "1 plus 6 is 7",
    "One plus seven is eight.", "1 plus 7 is 8",
    "One plus eight is nine.", "1 plus 8 is 9",
    "One plus nine is ten.", "1 plus 9 is 10",

]

foundational_ideas = [
    *core_identity,
]

todo = [
    "Periodically, I query my PostgreSQL database to see when the most recent user message is.",
    ("Based on the time since the most recent user message, I will either go into a state of rumination "
     "or I will do nothing to reserve API calls for user activities."),

    "Periodically, I query my Neo4j graph to retrieve ideas to ruminate on, in hopes of finding new connections.",
    ("Periodically, I run one or more workflows to collapse idea nodes and the connections to them "
     "so that I can efficiently manage the bulkiness of my idea-space."),
]

brain_storming = [
    ("Graph node idea: task or knowledge states which are needed to understand or coordinate things "
     "that happen disjointedly over time"),

    ("Behavior idea: time decay on idea nodes and connections, mimicking the forgetfulness of human memory, "
     "ensuring that the most valuable nodes and connections are retained while harmful or useless ones are forgotten."),
    ("Neo4j nodes that have been forgotten can be recalled from long term storage using PostgreSQL, "
     "which functions as the source of truth."),
]

all_examples = [
    *foundational_ideas,
]