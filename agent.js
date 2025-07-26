import {ChatOpenAI} from "@langchain/openai";
import {createReactAgent} from '@langchain/langgraph/prebuilt';
import {RecursiveCharacterTextSplitter} from '@langchain/textsplitters' //chunk spliter
import { OpenAIEmbeddings } from "@langchain/openai";
import {Document} from '@langchain/core/documents'
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { tool } from "@langchain/core/tools";
import {z} from 'zod'


import data from './data.js'

import dotenv from 'dotenv';

// Load .env into process.env
dotenv.config();

const video1=data[0]
const docs = [new Document({
    pageContent: video1.transcript,
    metadata: {
        video_id: video1.video_id
    }
})]

//instantiating the chat model
const model = new ChatOpenAI({ 
  model: "gpt-4o",
  apiKey: process.env.OPENAI_API_KEY
});



//instantiating the OpenAI Embeddin model
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

//instantiate local memory
const vectorStore = new MemoryVectorStore(embeddings);

//defining the splitting 
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});

//splitting into chunks
const chunks = await splitter.splitDocuments(docs)

//indexing
await vectorStore.addDocuments(chunks)

//retrieve the most relevant chunks
const retriever = await vectorStore.similaritySearch ('What was the finish time of Norris?', 1)


//retrieveal tool
const retrieveTool = tool( async ({query}) => {
console.log('Retrieving docs for the quey--------------');
//console.log(query)

const retrievedDocs = await vectorStore.similaritySearch(query, 3)
const serializedDocs = retrievedDocs.map ((doc)=> doc.pageContent).join('\n');

return serializedDocs;

},
{ //metadata dor the LLM
    name: 'retrieve',
    description: 'Retrieve the most relevant chunks of text from the transcript of a video',
    schema: z.object({ //zos schema for the LLM
        query: z.string()
    })
})

//creating the ReAct Agent
const agent = createReactAgent({
    llm: model,
    tools: [ retrieveTool],
});

const results =  await agent.invoke({
    messages:[{
        role: "user",
        content: 'Who was 200 of a seconds away?'
    }]
})

console.log(results)