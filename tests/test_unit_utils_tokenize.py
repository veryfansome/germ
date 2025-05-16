from germ.utils.tokenize import naive_tokenize


def test_naive_tokenize_hello_world_exclamation():
    text = "Hello  world!"
    expected = ['Hello', 'world', '!']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_hello_question():
    text = "Hello?"
    expected = ['Hello', '?']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_hello_im_c3po():
    text = "Hello, I'm C-3PO. Human-cyborg relations."
    expected = ['Hello', ',', 'I', "'m", 'C-3PO', '.', 'Human-cyborg', 'relations', '.']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_so_then_i_said():
    text = 'So then I said, "Hello world!"'
    expected = ['So', 'then', 'I', 'said', ',', '"', 'Hello', 'world', '!', '"']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_four_apostrophes_and_quotation_marks():
    text = "'''' \"\"\"\""
    expected = ["''''", '""""']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_hell_carry_that():
    text = "He'll have to carry that?!"
    expected = ['He', "'ll", 'have', 'to', 'carry', 'that', '?!']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_yes_that_makes():
    text = "Yes, that makes -what are you doing?"
    expected = ['Yes', ',', 'that', 'makes', '-', 'what', 'are', 'you', 'doing', '?']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_sure_i_can_did():
    text = "Sure, I can- did you see that??"
    expected = ['Sure', ',', 'I', 'can', '-', 'did', 'you', 'see', 'that', '??']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_attention_squirrel():
    text = "I don't think I have a short attention—squirrel!!"
    expected = ['I', 'do', "n't", 'think', 'I', 'have', 'a', 'short', 'attention', '—', 'squirrel', '!!']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_localhost():
    text = "The localhost address is 127.0.0.1."
    expected = ['The', 'localhost', 'address', 'is', '127.0.0.1', '.']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_tab_dash_foo():
    text = "\t- This is a Foo."
    expected = ['-', 'This', 'is', 'a', 'Foo', '.']
    assert naive_tokenize(text) == expected


def test_naive_tokenize_this_is_a_thing():
    text = "This is a thing -- this is another thing."
    expected = ['This', 'is', 'a', 'thing', '--', 'this', 'is', 'another', 'thing', '.']
    assert naive_tokenize(text) == expected
