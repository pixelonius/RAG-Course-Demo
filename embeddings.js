import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


//instantiating the OpenAI Embeddin model
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

//instantiate local memory
export const vectorStore = new MemoryVectorStore(embeddings);
export const addDocumentsToVectorStore = async (data)=>{
    const docs = [new Document({
        pageContent: data.transcript,
        metadata: {
            video_id: data.video_id
        }
    })]
    
    //defining the splitting 
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    });
    
    //splitting into chunks
    const chunks = await splitter.splitDocuments(docs)
    
    //indexing
    await vectorStore.addDocuments(chunks)
}