# chatbot.py

this is a little bot that reads all messages in wx team chat,
if a message is numeric, between 1 and 6000,
the bot assumes that it is a game level number,
it looks up all puzzle words that are valid for that level and
replies to the person with a list of answers.

this was put together by looking at mitmproxy captures and some debugging.
`config' object is missing some values - those are marked with FIXME
also missing all json data files from the ipa/apk.
