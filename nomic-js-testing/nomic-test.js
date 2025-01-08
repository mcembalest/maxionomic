import express from 'express';
import { embed, AtlasDataset, AtlasViewer } from '@nomic-ai/atlas';
// atlas = import("https://cdn.skypack.dev/@nomic-ai/atlas@next");
const anonViewer = new AtlasViewer({});
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

app.post('/embed', async (req, res) => {
  try {
    const { text } = req.body;
    const strings = text.split('\n').filter(str => str.trim());
    const embeddings = await embed(strings);
    res.json({ embeddings });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/dataset_by_id/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const dataset = new AtlasDataset(id);
    const attributes = await dataset.fetchAttributes();
    res.json(attributes);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/dataset_by_slug/:slug', async (req, res) => {
  try {
    const { slug } = req.params;
    const dataset = new AtlasDataset(slug, anonViewer).withLoadedAttributes();
    const attributes = await dataset.fetchAttributes();
    res.json(attributes);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});