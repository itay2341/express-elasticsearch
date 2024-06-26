const express = require('express');
const { Client } = require('@elastic/elasticsearch');
const ort = require('onnxruntime-node');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const app = express();
const port = 3000;

app.use(bodyParser.json());

// Initialize Elasticsearch client
const client = new Client({ node: 'http://localhost:9200' });

// Path to the ONNX model file
const modelPath = './all-MiniLM-L6-v2.onnx';

// Elasticsearch index name and mapping
const indexName = 'all_products';
const indexMapping = {
    mappings: {
        properties: {
            ProductID: { type: 'long' },
            ProductName: { type: 'text' },
            Price: { type: 'float' },
            Shipping: { type: 'text' },
            LinkToPost: { type: 'text' },
            Images: { type: 'text' },
            SellerLink: { type: 'text' },
            SellerName: { type: 'text' },
            Condition: { type: 'text' },
            ConditionDescription: { type: 'text' },
            Size: { type: 'text' },
            Description: { type: 'text' },
            NameVector: { type: 'dense_vector', dims: 384 } // Vector for the 'name' property
        }
    }
};

// Function to create Elasticsearch index
async function createIndex() {
    const exists = await client.indices.exists({ index: indexName });
    if (exists) {
        await client.indices.delete({ index: indexName });
    }
    await client.indices.create({ index: indexName, body: indexMapping });
}

// Function to generate embeddings for a given text
async function getEmbedding(session, text) {
    const { AutoTokenizer } = await import('@xenova/transformers');
    const tokenizer = await AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');

    // Ensure text is a string
    if (typeof text !== 'string') {
        text = String(text);
    }

    const inputs = tokenizer(text, { return_tensors: 'np', max_length: 128, padding: 'max_length', truncation: true });

    // Check for required properties in tokenized inputs
    if (!inputs.input_ids || !inputs.attention_mask || !inputs.token_type_ids) {
        throw new Error('Tokenized inputs are missing required properties.');
    }

    const inputIdsData = BigInt64Array.from(inputs.input_ids.data);
    const attentionMaskData = BigInt64Array.from(inputs.attention_mask.data);
    const tokenTypeIdsData = BigInt64Array.from(inputs.token_type_ids.data);

    const inputIds = new ort.Tensor('int64', inputIdsData, [1, inputs.input_ids.data.length]);
    const attentionMask = new ort.Tensor('int64', attentionMaskData, [1, inputs.attention_mask.data.length]);
    const tokenTypeIds = new ort.Tensor('int64', tokenTypeIdsData, [1, inputs.token_type_ids.data.length]);

    const feeds = { input_ids: inputIds, attention_mask: attentionMask, token_type_ids: tokenTypeIds };
    const results = await session.run(feeds);

    if (!results['last_hidden_state']) {
        throw new Error('Missing last_hidden_state in model output.');
    }

    const embeddings = results['last_hidden_state'].data;
    const numTokens = inputs.attention_mask.data.reduce((sum, x) => sum + Number(x), 0);

    if (numTokens === 0) {
        throw new Error('Number of tokens is zero, cannot divide by zero.');
    }

    const pooledEmbedding = new Float32Array(384);

    // Average pooling of embeddings
    for (let i = 0; i < embeddings.length; i += 384) {
        for (let j = 0; j < 384; j++) {
            pooledEmbedding[j] += embeddings[i + j] / numTokens;
        }
    }

    return pooledEmbedding;
}

// Function to index documents into Elasticsearch
async function indexDocuments(session) {
    for (const file of fs.readdirSync(path.join(__dirname, 'data'))) {
        const jsonData = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', file), 'utf8'));
        for (const [i, item] of jsonData.entries()) {
            try {
                if (!item.name) {
                    throw new Error('Name field is missing');
                }

                const nameVector = await getEmbedding(session, item.name);

                if (nameVector.length !== 384) {
                    throw new Error(`Vector dimension mismatch for document ID ${i + 1}`);
                }

                const document = {
                    ProductID: i + 1,
                    ProductName: item.name,
                    Price: parseFloat(item.price.replace('$', '').replace(',', '')),
                    Shipping: item.shipping,
                    LinkToPost: item.linkToPost,
                    Images: item.images,
                    SellerLink: item.sellerLink,
                    SellerName: item.sellerName,
                    Condition: item.condition,
                    ConditionDescription: item.conditionDescription,
                    Size: item.size,
                    Description: item.description,
                    NameVector: Array.from(nameVector)
                };

                await client.index({ index: indexName, id: i + 1, body: document });
            } catch (e) {
                console.error(`Error indexing document ID ${i + 1}: ${e}`);
            }
        }
    }
}

// Function to perform vector similarity search
async function searchByNameVector(queryVector) {
    try {
        const response = await client.search({
            index: indexName,
            size: 5,
            query: {
                script_score: {
                    query: {
                        match_all: {}
                    },
                    script: {
                        source: "cosineSimilarity(params.query_vector, 'NameVector') + 1.0",
                        params: {
                            query_vector: queryVector
                        }
                    }
                }
            }
        });
        return response.hits.hits;
    } catch (error) {
        console.error(`Error searching by name vector: ${error}`);
        return [];
    }
}

// Route to index data into Elasticsearch
app.post('/enter-data', async (req, res) => {
    try {
        await createIndex();
        const session = await ort.InferenceSession.create(modelPath);
        await indexDocuments(session);
        res.status(200).send({ message: 'Data has been indexed successfully' });
    } catch (error) {
        res.status(500).send({ error: error.message });
    }
});

// Route to perform vector similarity search
app.post('/vector-search', async (req, res) => {
    const { queryText } = req.body;
    try {
        const session = await ort.InferenceSession.create(modelPath);
        const exampleQueryVector = await getEmbedding(session, queryText);
        const response = await searchByNameVector(Array.from(exampleQueryVector));
        res.status(200).send(response);
    } catch (error) {
        res.status(500).send({ error: error.message });
    }
});

// Route to perform text-based fuzzy search on ProductName and Description
app.post('/text-search', async (req, res) => {
    const { nameQueryText, descQueryText } = req.body;

    try {
        const response = await client.search({
            index: indexName,
            body: {
                query: {
                    bool: {
                        should: [
                            {
                                match: {
                                    ProductName: {
                                        query: nameQueryText,
                                        fuzziness: 'AUTO'
                                    }
                                }
                            },
                            {
                                match: {
                                    Description: {
                                        query: descQueryText,
                                        fuzziness: 'AUTO'
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        });

        res.status(200).send(response);
    } catch (error) {
        res.status(500).send({ error: error.message });
    }
});

// Start the Express server
app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
