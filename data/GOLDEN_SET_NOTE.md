# Golden Evaluation Set Documentation (@AppleSupport)

## Purpose

The golden evaluation set is used exclusively for evaluating the
@AppleSupport support agent. It is kept separate from the RAG corpus.

## Construction Methodology

The working dataset sample contained a limited number of usable
@AppleSupport customer-support interactions. Therefore, a 200-example
evaluation set was constructed from recurring issue patterns observed
in the dataset and the project's predefined @AppleSupport intent taxonomy.

The evaluation set contains diverse synthetic customer phrasings rather
than claiming to be 200 raw historical tweets.

Examples were generated across six mutually exclusive intents:

- battery_drain
- software_bugs
- account_authentication
- hardware_repair
- billing_subscriptions
- general_inquiry

Each example was reviewed for:

1. Primary intent
2. Expected escalation decision
3. Reference support response
4. Response-quality rating

## Separation From RAG

The golden evaluation examples are never added to the RAG index.

The RAG index is built only from historical @AppleSupport
customer-support interactions extracted from the source dataset.

This separation prevents evaluation examples from being retrieved
during evaluation.

## Limitations

The golden set is synthetic rather than 200 independent historical
tweets. Therefore, results should be interpreted as performance on a
controlled evaluation set rather than as a direct estimate of
real-world production performance.

The set may also underrepresent issue types that are rare in the
available @AppleSupport sample.