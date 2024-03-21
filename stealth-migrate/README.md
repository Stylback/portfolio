Migrating Data from the Stealth Application
==============
**Authors:** *JONAS STYLBÄCK*

# Background

In April of 2023 the social news aggregator Reddit annonced changes to their application programming interface (API). Previously free, Reddit would now charge third-party applications a fee for each use of the API. While small, the cost of using the API would quickly pile up for some popular third-party applications ($20 million, in the case of Apollo). These changes was seen as the death for third-party Reddit applications and as such caused an [outrage in the community](https://en.wikipedia.org/wiki/2023_Reddit_API_controversy).

I had long been using [Stealth](https://gitlab.com/cosmosapps/stealth), developed by CosmoApps, to interact with Reddit. As a result of these changes, Stealth annonced the cessation of continued development and gave their users some time to export their contents in the form of a JSON file. I had a number of saved posts and comments in the export that I wanted to convert to a human-redable format, as such I created this script to extract the items of interest.

# The Script

The scripts takes the archive in JSON format, parses it to identify items of interests and saves them in a structured plain-text format. While simple, `"name": "value"` pairs relating to saved posts and comments was structured in HTML format which would require an additional level parsing. As an example, here is a saved comment in JSON format on a discussion thread relating to cross-referencing of critical vulnerability detection.

```json
{
    "total_awards": 0,
    "link_id": "t3_1455z9n",
    "author": "bitslammer",
    "score": "13",
    "body_html": "<div class=\"md\"><p>NIST offers an API which will allow you to automate this: <a href=\"https://nvd.nist.gov/developers/vulnerabilities\">https://nvd.nist.gov/developers/vulnerabilities</a></p>\n</div>",
    "edited": -1,
    "submitter": false,
    "stickied": false,
    "score_hidden": false,
    "permalink": "/r/cybersecurity/comments/1455z9n/anyone_know_any_good_website_to_quickly/jnj3zlq/",
    "id": "jnj3zlq",
    "created": 1686319310000,
    "controversiality": 0,
    "poster_type": "REGULAR",
    "link_title": "Anyone know any good website to quickly cross-reference CVE's that isn't the NIST website?",
    "link_permalink": "/r/cybersecurity/comments/1455z9n/anyone_know_any_good_website_to_quickly/",
    "link_author": "noahreeves446",
    "subreddit": "r/cybersecurity",
    "name": "t1_jnj3zlq",
    "time": 1686392070774
    },
```

The items of interest is the comment itself and any URLs within it. Below is the script output:

```
SAVED COMMENTS
---------
NIST offers an API which will allow you to automate this: https://nvd.nist.gov/developers/vulnerabilities
LINKS: https://nvd.nist.gov/developers/vulnerabilities
---------
```

# Experience Gained

I learned more about scripting in Python, parsing JSON and HTML data as well as using the [Beautiful Soup library](https://beautiful-soup-4.readthedocs.io/en/latest/). This was one of the first Python scripts I created for private use, unrelated to any course-works.