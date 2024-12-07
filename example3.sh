python3 autocoder.py  -i autocoder.py README.md -r "The README.md is out of date. Please review the code and update the README with the goals and details of the project. Mention any similar projects you know about and how they differ from this one. Mention that the proper way to store an API key is as an environment variable, with a fallback as a text file although that is widely seen as insecure. Also mention as of December 2024, a key from Hyperbolic.xyz is free and comes with \$10 in free credits, which would take literal weeks to blow through with this program even running it constantly. The author of the project is James Lemley (and has no connection to any AI service provider). As this is a personal non-work project, pull requests might be responded to or ignored. Oh and please add somewhere that that OpenAI's moat, if it ever existed, is gone. 
Oh, mention that incomplete chunks from the AI provider have been addressed through a continuation mechanism and system prompt, but when using models with small max token limits, continuation messages may be present in the final output. As always, a human should review the results with 'diff -u' before discarding the input files. The smallest max token api call is 8192 tokens which translates to a source code file of about 32KB. Models with larger max token parameters should scale. 

List a table of the models currently supported.

List a couple examples. A use case might be 'Use the argparse library to set options'. 

Please output a new README.md now." -o README.new.md -f 
