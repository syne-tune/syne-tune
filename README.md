---
title: README
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: etalab-2.0
language:
- fr
configs:
  - config_name: default
    data_files: votes.parquet
---
<style>
  @import url('https://fonts.googleapis.com/css2?family=Marianne:wght@300;400;500;700&display=swap');
  
  :root {
    --primary-color: #000091;
    --secondary-color: #6a6af4;
    --accent-color: #e1000f;
    --text-color: #1e1e1e;
    --light-bg: #f5f5fe;
    --border-radius: 6px;
  }
  
  @media (prefers-color-scheme: dark) {
    :root {
      --primary-color: #6a6af4;
      --secondary-color: #8989ff;
      --accent-color: #ff5c5c;
      --text-color: #e0e0e0;
      --light-bg: #252535;
    }
  }
  
  .container {
    font-family: 'Marianne', sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 15px;
    color: var(--text-color);
    line-height: 1.4;
  }
  
  @media (prefers-color-scheme: dark) {
    .container {
      color: var(--text-color);
    }
    
    a {
      color: var(--secondary-color);
    }
    
    code {
      background-color: #333;
      color: #f0f0f0;
    }
  }
  
  .logo {
    width: 200px;
    display: block;
    margin-left: 0;
    transition: transform 0.3s ease;
  }
  
  .logo:hover {
    transform: scale(1.05);
  }
  
  h1 {
    color: var(--primary-color);
    text-align: left;
    font-size: 2em;
    margin: 40px 0 30px;
    position: relative;
    padding-bottom: 15px;
  }
  
  h1::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 80px;
    height: 4px;
    background-color: var(--accent-color);
    border-radius: 2px;
  }
  
  .dataset-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0px;
    margin: 20px 0;
  }
  
  .dataset-card {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 25px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
  }
  
  .stats-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    margin: 25px 0;
  }
  
  .stat-card {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 20px;
    box-shadow: 0 3px 10px rgba(0, 0, 145, 0.1);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  
  .stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0, 0, 145, 0.15);
  }
  
  .stat-card .number {
    font-size: 2em;
    font-weight: 700;
    color: var(--primary-color);
    margin: 5px 0;
  }
  
  .stat-card .label {
    font-size: 0.9em;
    color: #555;
    font-weight: 500;
  }
  
  .dataset-metrics {
    background-color: var(--light-bg);
    padding: 15px;
    border-radius: var(--border-radius);
    margin-bottom: 20px;
    text-align: center;
    display: inline-block;
    min-width: 150px;
  }
  
  .dataset-metrics .number {
    font-size: 1.6em;
    font-weight: 700;
    color: var(--primary-color);
    margin: 4px 0;
  }
  
  .dataset-metrics .label {
    font-size: 0.9em;
    color: #555;
  }
  
  .metrics-container {
    display: flex;
    justify-content: flex-start;
    gap: 20px;
    flex-wrap: wrap;
    margin: 20px 0;
  }
  
  .video-container {
    box-shadow: 0 4px 12px rgba(0, 0, 145, 0.15);
    border-radius: var(--border-radius);
    overflow: hidden;
    width: 100%;
    max-width: 600px;
    margin: 20px auto 20px auto;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    padding-top: 0; 
  }
  
  .video-container video {
    width: 100%;
    display: block;
  }
  
  .video-container:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0, 0, 145, 0.2);
  }
  
  .highlight-box {
    background-color: var(--light-bg);
    padding: 20px;
    border-radius: var(--border-radius);
    margin: 25px 0;
  }
  
  .button {
    display: inline-block;
    background-color: var(--secondary-color);
    color: white !important;
    text-decoration: none;
    padding: 10px 20px;
    border-radius: var(--border-radius);
    font-weight: 500;
    transition: all 0.3s ease;
    margin: 5px;
    border: 1px solid var(--secondary-color);
  }
  
  .button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
  
  .button.secondary {
    background-color: #6A6AF4;
    color: white !important;
    border: 1px solid var(--primary-color);
  }
  
  .datasets-buttons {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0;
  }
  
  .contact-section {
    text-align: left;
    margin-top: 40px;
    padding: 20px;
    background-color: var(--light-bg);
    border-radius: var(--border-radius);
  }
  
  
  .data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
  }
  
  .data-table th {
    background-color: var(--light-bg);
    padding: 10px;
    text-align: left;
    color: var(--primary-color);
  }
  
  .data-table td {
    padding: 10px;
    border-bottom: 1px solid #eee;
  }
  
  .datasets-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 15px 0;
  }
  
  .datasets-grid .highlight-box {
    margin-top: 0;
    margin-bottom: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  
  .datasets-grid .highlight-box h3 {
    margin-top: 0;
    margin-bottom: 10px;
  }
  
  .datasets-grid .video-container {
    margin: 15px 0 0 0;
    max-width: 100%;
  }
  
  /* Style pour une vidéo plus élégante et compacte */
  .video-container.compact {
    max-width: 480px;
    margin: 25px auto;
    box-shadow: 0 3px 10px rgba(0, 0, 145, 0.1);
    border: 1px solid var(--light-bg);
  }
  
  .video-container.compact video {
    display: block;
    width: 100%;
    height: auto;
  }
</style>

<div class="container">
  <a href="https://comparia.beta.gouv.fr/">
    <img class="logo" src="https://github.com/user-attachments/assets/bd071ffd-1253-486d-ad18-9f5b371788b0" alt="compar:IA logo">
  </a>

# comparia-votes – the dataset of all preferences expressed by compar:IA users

## Origin of the data: what is compar:IA?

[Compar:IA](https://comparia.beta.gouv.fr/) is a conversational AI comparison tool (a "chatbot arena") developed within the French Ministry of Culture with a dual mission:
- Educate and raise awareness about model pluralism, cultural and linguistic biases, and the environmental issues of conversational AI.
- Improve French conversational AI by publishing French alignment datasets and building a ranking of French conversational AI models (in progress). 

The compar:IA comparator is developed as part of the State startup compar:IA (incubated by the [Atelier numérique](https://www.culture.gouv.fr/Thematiques/innovation-numerique/Aides-a-l-innovation-et-a-la-transformation-numerique/L-Atelier-numerique#:~:text=L'Atelier%20num%C3%A9rique%20est%20l,engager%20personnellement%20pour%20le%20r%C3%A9soudre.) and [AllIAnce](https://alliance.numerique.gouv.fr/)), integrated into the [beta.gouv.fr](beta.gouv.fr) program of the [Interministerial Digital Directorate (DINUM)](https://www.numerique.gouv.fr/dinum/), which helps public administrations build useful, simple, and easy-to-use digital services.
<div style="margin: 20px 0;">
  <a href="https://comparia.beta.gouv.fr/" class="button secondary">compar:IA platform website</a>
  <a href="https://github.com/betagouv/ComparIA" class="button secondary">compar:IA source code</a>
</div>

## Definition of a preference on compar:IA

After a full conversation on compar:IA, if the user has not rated a specific message, before revealing the models they can vote for one of the two models. Alternatively, the user can decide that both models gave answers of equal quality. 

After that, the user can also select qualifiers to rate model performance over the whole conversation. 

<video controls autoplay loop muted playsinline src="https://cdn-uploads.huggingface.co/production/uploads/649d986a474bf415c03b772c/Fv-aTZYUKDsPwS5HwNbX3.mp4"></video>


## Dataset content 

In total on compar:IA, more than 100k conversations have taken place. You can find them all in this dataset – [comparia-conversations.](https://huggingface.co/datasets/ministere-culture/comparia-conversations)

Among these conversations, users voted more than 30k times on conversations. These conversations and the corresponding votes are available in this dataset. The conversations are mostly **in French** and reflect **real, unconstrained uses**.


## Columns of the comparia-votes dataset

<table class="data-table">
  <tr>
    <th>Column</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>id</code></td>
    <td>Unique identifier for each entry in the dataset</td>
  </tr>
  <tr>
    <td><code>timestamp</code></td>
    <td>Conversation timestamp</td>
  </tr>
  <tr>
    <td><code>model_a_name</code></td>
    <td>Name of the first model</td>
  </tr>
  <tr>
    <td><code>model_b_name</code></td>
    <td>Name of the second model</td>
  </tr>
  <tr>
    <td><code>model_pair_name</code></td>
    <td>Set representation of the two compared models</td>
  </tr>
  <tr>
    <td><code>chosen_model_name</code></td>
    <td>Name of the model the user voted for</td>
  </tr>
  <tr>
    <td><code>opening_msg</code></td>
    <td>First message sent by the user</td>
  </tr>
  <tr>
    <td><code>both_equal</code></td>
    <td>Indicates whether the user judged the two models to be equal</td>
  </tr>
  <tr>
    <td><code>conversation_a</code></td>
    <td>Full structure of the conversation with the first model</td>
  </tr>
  <tr>
    <td><code>conversation_b</code></td>
    <td>Full structure of the conversation with the second model</td>
  </tr>
  <tr>
    <td><code>conv_turns</code></td>
    <td>Number of dialogue turns in the conversation</td>
  </tr>
  <tr>
    <td><code>selected_category</code></td>
    <td>Prompt suggestion category chosen by the user (if they chose a suggested prompt)</td>
  </tr>
  <tr>
    <td><code>is_unedited_prompt</code></td>
    <td>Indicates whether the suggested prompt was used as-is</td>
  </tr>
  <tr>
    <td><code>conversation_pair_id</code></td>
    <td>Unique identifier for the pair of conversations</td>
  </tr>
  <tr>
    <td><code>session_hash</code></td>
    <td>User session identifier</td>
  </tr>
  <tr>
    <td><code>visitor_id</code></td>
    <td>Unique anonymized identifier for the user</td>
  </tr>
  <tr>
    <td><code>conv_comments_a</code></td>
    <td>Comments on the conversation with the first model</td>
  </tr>
  <tr>
    <td><code>conv_comments_b</code></td>
    <td>Comments on the conversation with the second model</td>
  </tr>
  <tr>
    <td><code>conv_useful_a</code></td>
    <td>Indicates whether the conversation with the first model was judged useful</td>
  </tr>
  <tr>
    <td><code>conv_useful_b</code></td>
    <td>Indicates whether the conversation with the second model was judged useful</td>
  </tr>
  <tr>
    <td><code>conv_creative_a</code></td>
    <td>Indicates whether the first model’s answer was judged creative</td>
  </tr>
  <tr>
    <td><code>conv_creative_b</code></td>
    <td>Indicates whether the second model’s answer was judged creative</td>
  </tr>
  <tr>
    <td><code>conv_clear_formatting_a</code></td>
    <td>Indicates whether the first model’s formatting was clear</td>
  </tr>
  <tr>
    <td><code>conv_clear_formatting_b</code></td>
    <td>Indicates whether the second model’s formatting was clear</td>
  </tr>
  <tr>
    <td><code>conv_incorrect_a</code></td>
    <td>Indicates whether the first model’s answer contained incorrect information</td>
  </tr>
  <tr>
    <td><code>conv_incorrect_b</code></td>
    <td>Indicates whether the second model’s answer contained incorrect information</td>
  </tr>
  <tr>
    <td><code>conv_superficial_a</code></td>
    <td>Indicates whether the first model’s answer was judged superficial</td>
  </tr>
  <tr>
    <td><code>conv_superficial_b</code></td>
    <td>Indicates whether the second model’s answer was judged superficial</td>
  </tr>
  <tr>
    <td><code>conv_instructions_not_followed_a</code></td>
    <td>Indicates whether the first model did not follow instructions</td>
  </tr>
  <tr>
    <td><code>conv_instructions_not_followed_b</code></td>
    <td>Indicates whether the second model did not follow instructions</td>
  </tr>
  <tr>
    <td><code>system_prompt_b</code></td>
    <td>System instruction provided to the second model</td>
  </tr>
  <tr>
    <td><code>system_prompt_a</code></td>
    <td>System instruction provided to the first model</td>
  </tr>
  <tr>
    <td><code>conv_complete_a</code></td>
    <td> - </td>
  </tr>
  <tr>
    <td><code>conv_complete_b</code></td>
    <td> - </td>
  </tr>
</table>

## Purpose of this dataset

We make this dataset available to model developers and to the AI and social science research community to support progress in the following areas:
- Training and alignment of conversational language models, especially in French
- Human–machine interactions and the specific behaviors involved in conversational AI systems
- Improving LLM evaluation methods
- AI safety and content moderation

If you use the compar:IA dataset, we would love to hear about your use cases and feedback. Your feedback will help us improve the reuse experience. You can contact us at <a href="mailto:contact@comparia.beta.gouv.fr">contact@comparia.beta.gouv.fr</a>.

## Data post-processing

User consent is collected through the “Terms of use” section on the site. A detection of personally identifiable information (PII) was carried out (results are shown in the 'contains_pii' column of the dataset), and conversations containing such information were anonymized. However, we do not apply any filtering or processing of potentially toxic or hateful content, to allow researchers to study safety issues related to LLM use in real-world contexts.

## Licenses

Subject to third-party claims regarding model-generated results, we make the dataset available under the Etalab 2.0 open license. It is the responsibility of users to ensure that their use of the dataset complies with applicable regulations, in particular regarding personal data protection and the terms of use of the different model providers.

## Other compar:IA datasets

<div class="datasets-grid">

  <div class="highlight-box">
    <h3>comparIA-conversations</h3>
    <p>Dataset containing all questions asked and answers received on the compar:IA platform.</p>
      <image src="https://cdn-uploads.huggingface.co/production/uploads/649d986a474bf415c03b772c/LUYr4vyM1eeHGQ5JSHJQR.png"></image>
    <div class="datasets-buttons">
      <a href="https://huggingface.co/datasets/ministere-culture/comparia-conversations" class="button secondary">Explore comparIA-conversations</a>
    </div>
  </div>
  
  <div class="highlight-box">
    <h3>comparIA-reactions</h3>
    <p>Dataset collecting user reactions to compar:IA at the message level. It reflects preferences expressed throughout conversations, message by message.</p>
    <div class="video-container">
      <video controls autoplay loop muted playsinline src="https://cdn-uploads.huggingface.co/production/uploads/649d986a474bf415c03b772c/ncldPIO_bTesSd8bqcjqn.mp4"></video>
    </div>
    <div class="datasets-buttons">
      <a href="https://huggingface.co/datasets/ministere-culture/comparia-reactions" class="button secondary">Explore comparIA-reactions</a>
    </div>
  </div>


</div>



<div class="contact-section">
  <h3>Reporting sensitive data</h3>
  <p>If you find a line in the dataset that you think contains PII or sensitive data, please let us know via  <a href="https://adtk8x51mbw.eu.typeform.com/to/B49aloXZ">this short form</a>.</p>
  
  <h3>Contact</h3>
  <p>For any question or request for information, contact <a href="mailto:contact@comparia.beta.gouv.fr">contact@comparia.beta.gouv.fr</a></p>
  
  <div style="margin-top: 30px;">
    <a href="https://beta.gouv.fr">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/649d986a474bf415c03b772c/Zk4YiqgKu9sm5ydQ7fhSq.png" alt="Logo of the Ministry, beta.gouv and Atelier numérique" style="max-width: 400px;">
    </a>
  </div>
</div>

</div>

<div align="center">

<br />
<a href="https://digitalpublicgoods.net/r/comparia" target="_blank" rel="noopener noreferrer"><img src="https://github.com/DPGAlliance/dpg-resources/blob/main/docs/assets/dpg-badge.png?raw=true" width="100" alt="Digital Public Goods Badge"></a>

</div>
