import pytest

from app.model import train

# A tiny two-domain dataset so tests run in seconds without downloading CLINC150.
ROWS = [
    ("what is my account balance", "balance"),
    ("how much money do i have in checking", "balance"),
    ("show me my savings balance", "balance"),
    ("tell me the balance on my account", "balance"),
    ("send 50 dollars to my brother", "transfer"),
    ("move money from savings to checking", "transfer"),
    ("transfer 200 to john", "transfer"),
    ("i want to send money to my mom", "transfer"),
    ("play some jazz music", "play_music"),
    ("put on my workout playlist", "play_music"),
    ("play a song by adele", "play_music"),
    ("start playing rock music", "play_music"),
    ("turn the volume up", "volume_up"),
    ("make it louder", "volume_up"),
    ("increase the volume please", "volume_up"),
    ("louder please, turn it up", "volume_up"),
]
INTENT_TO_DOMAIN = {"balance": "banking", "transfer": "banking", "play_music": "media", "volume_up": "media"}


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train(ROWS * 3, INTENT_TO_DOMAIN, out, threshold=0.05)
    return out
