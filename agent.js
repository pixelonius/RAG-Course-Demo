import {ChatOpenAI} from "@langchain/openai";
import {createReactAgent} from '@langchain/langgraph/prebuilt';
import {RecursiveCharacterTextSplitter} from '@langchain/textsplitters' //chunk spliter
import { OpenAIEmbeddings } from "@langchain/openai";
import {Document} from '@langchain/core/documents'

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

//creating the ReAct Agent
const agent = createReactAgent({
    llm: model,
    tools: [],
});

//instantiating the OpenAI Embeddin model
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

//splitting the video into chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});

const chunks = await splitter.splitDocuments(docs)

console.log(chunks)

//embed the chunks






// const result = await agent.invoke({
//   messages: [
//     {
//       role: "user",
//       content: "What is the capital of the Romania?",
//     },
//   ],
// });

// console.log(result.messages.at(-1)?.content)