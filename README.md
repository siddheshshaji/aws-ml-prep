# aws-ml-prep

aws ml speciality

https://github.com/FabG/ml-aws-specialty-lab/blob/master/aws-notes/aws-machine-learning-cheat-sheet-112020.pdf

AWS REKOGNITION

-  deep learning for various image-based workloads such as image classification, object detection, detection of text in image, facial recognition, sentiment, and most recently, public safety
￼
 
- lower levels of cnn architecture are used for detecting simple features like edges, corners, shapes, etc. while the higher levels are used for detecting complex featues like faces, etc. so in transfer learning, only the higher levels are trained.
- for object detection >> similar to object classification, but a rectangular box is drawn around the object in question >> single-shot detector (SSD), R-CNN or Faster R-CNN, and YOLO v4. 
- semantic segmentation >> segments the object of interest in an image by classifying whether or not an object belongs in a given pixel
- Image Labeling, Custom Image Labeling, Face Detection and Search, People Paths, Text Detection, Celebrity Detection, Personal Protective Equipment (PPE)
- Look out for key phrases like “without any prior machine learning/deep learning knowledge” or “cost effective” or any of the use cases just described to think of Amazon Rekognition as the solution. If the question contains a phrase like “custom model,” unless it has to do with image labeling, usually Amazon Rekognition is not the answer.
- Amazon Rekognition operates on both static images and stored videos. Image operations are synchronous whereas video operations are asynchronous. This means that when you ask Amazon Rekognition to process a video, once the job is completed, Amazon Rekognition will notify you using Amazon SNS by publishing to an SNS topic. You then have to call a Get* API to access the outputs. For synchronous API calls, you will get the answer right away.
- Amazon Rekognition does not support video for all operations; for example, the PPE detection APIs only support images. Likewise, the people pathing use case is only available for video and not for images.
- For object detection in an image, you simply need to pass in the location of the image (in JPEG or PNG) in Amazon S3 or a byte-encoded image input
- By contrast, for a video job, you cannot pass in bytes but must pass in the location of a video stored in Amazon S3. The API is StartLabelDetection and you also need to pass in an SNS topic for Amazon Rekognition to push a notification to, once it completes the video labeling task
- A key benefit of Amazon Rekognition Video is that you can work with streaming videos. Amazon Rekognition can ingest streaming videos directly from Amazon Kinesis Video streams, process the videos, and publish the outputs to Amazon Kinesis Data Streams for stream processing.
- If you are looking to build a scalable image or video analytics workflow, consider using tools like the AWS Lambda function to make Amazon Rekognition API calls in a serverless manner. You may also consider using Amazon SQS to queue your incoming data to prevent throttling of Amazon Rekognition APIs
- Once your collection is created, consider using a tool like Kinesis Video Streams to first ingest the video stream and a Kinesis Data Stream as the output data stream. Rekognition Video can then process the incoming video stream using the CreateStreamProcessor API, passing the Kinesis Video stream as input. The outputs of the analysis will be published to Kinesis Data Streams. From Kinesis Data Streams you can use AWS Lambda as a consumer to publish the outputs to S3 or to a key-value store such as Amazon DynamoDB
- rekognition faqs: https://aws.amazon.com/rekognition/faqs/


AWS TEXTRACT

- Amazon Textract is an AI service that allows you to quickly extract intelligence from documents such as financial reports, medical records, tax forms, and university application forms beyond simple optical character recognition (OCR).
- Note that Textract is used for extracting forms, tables, and text from PDFs or images. It does not do document classification, sentiment analysis, or entity recognition on those documents themselves. That is done by a different service called Amazon Comprehend.
- works in async(longer docs like pdfs, forms larger than a single page, etc.) and sync(images, etc.) formats
- You can use a synchronous API such as DetectDocumentText or AnalyzeDocument to return a JSON output containing the detected or analyzed text.
- async jobs needs an sns topic registered which will notify the subscriber about the completion of the job.
- consider a form such as an application that contains a Name field, with the name John Smith. Now, if the service simply returned the outputs as Name, John Smith, that would not be very useful to someone trying to parse the document downstream. Amazon Textract instead returns the text as a key-value pair, allowing the user to seamlessly ingest these outputs into a key-value database store that you may use to query later
-  For documents in PDF form or documents that are larger than a single page, use the async APIs StartDocumentAnalysis and StartDocumentTextDetection. Since detecting text in large documents can take some time, Amazon Textract will process your documents behind the scenes and publish the Completion status to an SNS topic. A subscriber to this topic will be subsequently notified that the job is complete and can view the outputs by calling the GetDocumentAnalysis or GetDocumentTextDetection API.
- Repeatedly calling Textract APIs can result in a throttling exception called ProvisionedThroughputExceededException if the transactions per second (TPS) exceed the maximally allowed value. In that case, specify an automatic retry using the Config utility with the AWS SDK. Amazon Textract will automatically retry jobs a certain number of times before failing. Generally, you can set this value to 5.
- Amazon Augmented AI (Amazon A2I) can directly integrate with the Textract document analysis API to send documents for human review based on a particular threshold condition such as low confidence on the detected text
- faqs: https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html#limits_textract

AWS TRANSCRIBE

- Used for voice to text applications
- Stream and Batch Mode, Multiple Language Support, Multiple Language Transcription, Job Queuing, Custom Vocabulary and Filtering, Automatic Content Redaction, Language Identification, Speaker Identification
- Automatic Content Redaction  If your audio includes personally identifiable information (PII), Transcribe gives you the option to redact it from the transcribed output or provide both unredacted and redacted scripts. This information may include entities such as account numbers, credit card numbers, names, U.S. phone numbers, and U.S. Social Security numbers. Note that this feature is only available in English
- Know the difference between a custom model with Amazon Comprehend (which we will cover later) versus Amazon Transcribe. The Transcribe custom language model is only applicable for audio transcription use cases. Comprehend Custom is for document classification and entity extraction, among other uses.
- Furthermore, this domain has highly specialized vocabulary that exceeds the 50 KB limit for custom vocabulary. Your AWS Solutions Architect recommends that you build a custom model to improve the transcription accuracy. Amazon Transcribe now lets you build a custom language model simply by providing your text as an input. Transcribe will build the model, and then you can use this model instead for your domain-specific transcriptions. Furthermore, you can provide a training dataset consisting of your text, and a test dataset containing a sample of audio transcripts.
- Amazon Transcribe Medical is an ASR service that enables you to transcribe medical audio such as physician dictation, patient-to-physician conversations, and telemedicine
- Remember, not all AI services have a medical specialty. Among the ones that do are Comprehend and Transcribe. You may get a question on the test that requires custom transcription but the answers may include nonexistent services like Translate Medical or Textract Medical. Those are immediately incorrect, allowing you to narrow down your answers.

AWS TRANSLATE

- A text translation service that uses advanced state-of-the-art deep learning to provide high-quality translations to customers without any deep learning experience, with a pay-for-what-you-use pricing structure. As the name suggests, the main use case for Amazon Translate is to translate text
- As always, know what a service is and what it is not. Amazon Translate does not translate directly from voice, such as calls. You can, however, chain Amazon Transcribe with Amazon Translate to make this flow work
- Remember that Translate does not let you build your own custom translation models; it uses models trained by Amazon. 

AWS POLLY

- uses Speech Synthesis Markup Language syntax to convert text to speech
- text to speech tts and neural tts(ntts)


AWS LEX

- Amazon Lex is an AWS service, powered by natural language understanding (NLU) and automatic speech recognition (ASR), that allows users to build and deploy conversational interfaces for their applications.
- Amazon Lex is used to build and deploy the chat interface and responds to user intents
- With Amazon Lex, the backend actions can be performed by using an AWS Lambda function. For example, if your bot is designed to make an appointment at your local doctor’s office, you could have the Lambda function write to Amazon Relational Database Service (RDS) or Amazon Aurora or even a DynamoDB Appointments table. Likewise, if a customer wanted a reminder of their appointment, the bot could call a Lambda function to read from the table and return the appointment details.
- If you are building a bot using Lex, and if your bot is not performing well, try increasing the number of sample utterances. The more examples you provide, the better the model will be able to generalize to unseen utterances.
- Understand the differences between slots, utterances, and intents. Slots are configuration parameters, utterances are the actual sentences, and intents are the meanings behind them. In the pizza example, slots can be pizza size or individual toppings, the intent is to order a pizza, and utterances may be, “I want olives on my pizza,” “I want a small pizza,” or “Can I order a pizza.”
- Know which external tools Lex can integrate with, namely Facebook Messenger, Slack, and Twilio Short Message Service (SMS).

AWS KENDRA

- Kendra also uses NLP behind the scenes, but it is aimed at document search and question and answering (Q&A) as opposed to a general-purpose tool for NLP.
- Kendra comes in two pricing models: a developer edition and an enterprise edition. Both are pay-as-you-go, but the latter provides higher availability by operating in three availability zones (AZs), allows for more queries, and can ingest more documents and text
- Index: In order to search documents, first you need to index them. An index is an object that is managed by Kendra that carries some metadata about that document, such as when it was created and updated, the version, and custom fields such as date and number that you can modify as a user. Documents: These include the actual documents that Kendra will index. They may include frequently asked questions (FAQs) or purely unstructured documents such as HTML files, PDFs, plain-text or Microsoft Word documents, or Microsoft PowerPoint presentations. Data sources: You may be wondering if you need to manually index documents. The answer is no; you simply provide Kendra with a data source such as a Confluence server, Microsoft SharePoint, Salesforce sites, ServiceNow instances, or an Amazon S3 bucket, and Kendra will index your documents as well as synchronize the data source with the index to keep it relevant and updated. For a full list of the supported data sources for Amazon Kendra
- Know the difference between Comprehend and Kendra. Kendra also uses NLP behind the scenes, but it is aimed at document search and question and answering (Q&A) as opposed to a general-purpose tool for NLP
- You can now build an end-to-end FAQ chatbot using Lex and Kendra. Lex provides the front end to identify the user intent based on utterances, and it can call Kendra using KendraSearchIntent by passing in the intent as the input. Kendra can then search and return the most relevant results that are surfaced by the chatbot.
- You can now build an end-to-end FAQ chatbot using Lex and Kendra

AWS PERSONIALIZE

- Amazon Personalize is a machine learning service that allows businesses to rapidly develop personalized recommendation systems to provide a better customer experience to their end customers
- Based on: 
- User Data - This may include data about user’s age, location, income, gender, personal preferences, and other demographic information. Item Data  This is data about the actual products a company sells or is trying to recommend. User-Item Interaction Data  This is data about how the set of users have interacted with these items, such as whether they have purchased the items in the past, do they like or dislike these items, or have they provided reviews or ratings for these items
- clustering, collaborative filtering(Collaborative filtering decomposes large sparse matrices into smaller matrices (matrix factorization) to extract hidden or latent vectors for each user and each item. The dot product of these gives the final score, which determines whether or not to recommend an item. Although this is a highly popular technique, if you don’t have a lot of items, matrix factorization can often not work as well) and content-based filtering (conventional recommedation methods) based on the previously mentioned data formats
- One of the drawbacks of collaborative filtering is that it is time invariant; it doesn’t take into account a user’s purchase or session history
- hrnn-metadata(which uses an RNN to store user histories but also has the ability to incorporate user and item metadata, allows them to solve not only the temporal history problem, but simultaneously the cold start problem, which is the inability of a recommender to recommend products to completely new users) and multiarmed bandits(but at a high level, a MAB uses the concept of exploration-exploitation trade-off.) (contemporary popular ones)
- Amazon Personalize employs the concept of recipes, grouped into three types for a given use case. Recipes allow you to build recommender systems without any prior ML knowledge: User Personalization Recipes  These recipes come in three flavors. First, userpersonalization uses the user-item interaction data and tests different recommendation scenarios. It is the recommended personalization recipe and built using the explorationexploitation trade-off we discussed earlier. Second, popularity count recommends the most popular item among all your users and is a good baseline to compare other recipes against. Finally, there are legacy recipes that involve the HRNN and HRNN-meta models we discussed earlier. Ranking-Based Recipes This recipe also uses an HRNN but it also ranks the recommendations. Related Item Recipe  This is the collaborative filtering algorithm we described earlier.
- The performance of the model is based on evaluation metrics such as Precision at K and Mean Reciprocal Rank at K. Precision at K: Of the K items recommended, how many were actually relevant, divided by K. ■ Mean Reciprocal Rank at K: The mean of the reciprocal rank of the first recommendation out of K, where the mean is taken over all queries.
- Later in this chapter, we will cover Amazon SageMaker, which has a builtin algorithm called Factorization Machine. Understand that this is a supervised learning algorithm that works well when you have a small number of items compared to algorithms like HRNN that are ideally suited for large numbers of items (>100). In the test, if the question asks about Personalization on SageMaker, think Factorization Machine. This algorithm can be used for binary classification and regression problems, not multiclass classification
- https://www.udemy.com/course/aws-machine-learning/learn/lecture/29289090#notes

AWS FORECAST

- Amazon Forecast is an AI service that uses both statistical and deep learning–based algorithms to provide highly accurate forecasts.
- Deep-AR+ that uses a long short-term memory (LSTM) This is an extension of DeepAR we discussed earlier and trains a single model on many similar time series (>100s). It works by splitting your time series randomly into fixed-length “windows” called context length and aims to predict the future up to a length called the forecast horizon. By doing this over many epochs and different time series, DeepAR can learn common patterns across different time series to generate an accurate global model. DeepAR+ treats the context length as a hyperparameter in your model, allowing you to tune them to get better performance. Additionally, DeepAR accepts metadata about the item in the form of related time series or simply item metadata. So if you are forecasting sales, a related time series could be a time series of weather data or foot traffic data to your store. Although DeepAR+ can handle missing values in your data, if your data has many missing values, then your forecasts may suffer because the model is not able to learn useful patterns. expects input in json line format(gzip or parquet) (https://www.udemy.com/course/aws-machine-learning/learn/lecture/16586070#overview)
- ETS: ETS, or exponential smoothing, is a statistical algorithm that is useful for datasets with seasonality. It works by computing a weighted average of prior features, but instead of a constant weight, it applies an exponentially decaying function as the weighting parameter. ETS does not accept any metadata or related time series.
- Prophet: Prophet is useful when your time series has strong seasonal variations over many months/years and if you have detailed time series information. It is also useful when your data contains irregularities (such as spikes during holidays) or has missing values.
- CNN-QR: The convolutional neural network quantile regression algorithm also uses deep learning, but this time uses convolutional neural networks (CNN) over recurrent neural nets like DeepAR. It uses a concept called sequence-to-sequence learning, where a model is fed an input sequence and it generates a hidden representation of that sequence. This is called an encoder. That representation is then used to predict the output sequence using another network called the decoder. We don’t cover sequence-to-sequence models in great depth here, but it is useful to understand the distinction between this and DeepAR. DeepAR uses LSTMs whereas CNN-QR uses causal convolutional networks.
- Both these models accept metadata and related time series inputs. However, CNN-QR does not require the related time series to extend to the forecast horizon, but DeepAR does. Imagine you have a time series of item sales up to time t and you are trying to predict sales from time t + 1 to t + n into the future. If you are using weather data as your related time series, DeepAR requires you to have a weather forecast handy from time t + 1 to t + n in order to predict your future sales. CNN-QR does not have that requirement.
- If you only have a handful of time series, consider algorithms like ARIMA, ETS, or Prophet. Once you have hundreds of time series, only then consider DeepAR+ or CNN-QR.
- time series use a concept called backtesting, where a model is tested against historical data where you have ground truth.
- Amazon Forecast also provides a probabilistic forecast by providing you with quantiles such as p10, p50, or p90. It is useful to understand what these quantiles mean: ■ p10 means that your model predicts that the true value will be less than this value only 10 percent of the time. ■ p90 means that your model predicts that the true value will be less than this value 90 percent of the time.
- When should you use WAPE versus RMSE versus wQL loss? If your business will have an outsized impact for a few large mispredictions, then consider RMSE. If your business costs change based on whether your forecast under- or overpredicts, consider wQL loss. Otherwise, consider WAPE. In general, it is a good practice to look at your model performance against multiple metrics and visualize your predictions with different quantiles, such as p10, p50, and p90.


AWS COMPREHEND

- Amazon Comprehend provides a set of natural language processing–based APIs to pretrained and custom models that can extract insights from text. 
- Amazon Comprehend can analyze a document for the following characteristics: 1. Entities 2. Key phrases 3. Personally identifiable information (PII)—Data that could be used to identify an individual such as a name, address, or bank account number. In the previous example, “100 Main Street, Anytown, WA 98121” is PII data. 4. Language 5. Sentiment 6. Syntax
- Apart from these API calls to pretrained models, you can also train custom models on Amazon Comprehend using your own data.
- Custom document classification, Custom entity detection, Document topic modeling
- A customer collects incoming email in a JSON document and wants to use the subject line to redirect emails to the right department. The customer first organizes their data into a set of two-column CSV files; the first column is the department label, and the second column is the subject line. They can then use the console or the CreateDocumentClassifier API to start a custom training job. Amazon Comprehend uses between 10 and 20 percent of the training data for testing the final trained model and provides a classifier performance metric to help you determine if the model is trained well enough for your purposes. The customer can then analyze incoming emails by passing in the subject line to the hosted model endpoint and routing the email using the results obtained from this API call.

AWS CODEGURU

- Amazon CodeGuru uses program analysis and machine learning built from millions of lines of Java and Python code from the Amazon codebase to provide intelligent recommendations for improving code performance and quality
- CodeGuru consists of two main services: Reviewer and Profiler
- Amazon CodeGuru Reviewer proactively detects potential code defects and offers suggestions for improving your Java or Python code. CodeGuru Reviewer does not identify syntax errors (an IDE is a better way to do this), but it does suggest improvements related to AWS best practices, resource leak prevention, concurrency, sensitive information leak prevention, refactoring, input validation, and security analysis.
- Amazon CodeGuru Profiler collects runtime performance data from your live applications and provides recommendations on how to fine-tune performance.

AWS AUGMENTED AI

- Amazon Augmented AI (or A2I) is used to get a secondary human review of a low-confidence prediction from machine learning models.
- A human review workflow involves (1) defining a work team that will review predictions, and (2) using a UI template for providing instructions and the interface for humans to provide feedback (called the worker task template). A work team can be made up of a public workforce (powered by Amazon Mechanical Turk), a vendor-managed workforce, or your own private workforce. The worker task template displays your input data such as images or documents, and provides interactive tools to allow human reviewers to complete their task of reviewing the machine learning model’s prediction.
- A financial services company has a machine learning model that predicts whether a loan application is fraudulent or not. A recent mandate states that this company must review predictions of fraudulent loan applications by humans before making a decision on the loan. The company uses A2I to support automated machine learning by first calling the machine learning model endpoint and analyzing the confidence score. If the confidence score is less than 90 percent, the client triggers a human review loop in A2I and later analyze these results from humans from output files stored in Amazon S3

AWS SAGEMAKER

- Amazon SageMaker is an end-to-end machine learning platform that lets you build, train, tune, and deploy models at scale

￼

- Files stored inside the /home/ec2-user/SageMaker directory persist between notebook sessions (that is, when you turn the notebook instance off and on again). 
- Note that scheduling a notebook to be turned off during idle times is important to reduce costs; this can be done using lifecycle configuration scripts or via Lambda functions.
- Compared to notebook instances, SageMaker Studio launches containerized images that are used to run kernels for your notebooks.
- With GroundTruth, you can use a public workforce (Amazon Mechanical Turk), a private workforce, or a vendor company. You can optionally use automated data labeling for some task types, which uses active learning to train a model in parallel and decide which samples of data to send to human labelers. 
- Distributed Training SageMaker provides both model parallel and data parallel distributed training strategies. Let’s briefly discuss what these strategies are. Data parallel strategy in distributed training is where the dataset is split up across multiple processing nodes. Each node runs an epoch of training and shares results with other nodes before moving on to the next epoch. In model parallel training, the model is split up across multiple processing nodes. Each node carries a subset of the models and is responsible to run a subset of the transformations as decided by a pipeline execution schedule so that performance losses due to sequential computations are minimized.
- Endpoint Autoscaling  Dynamically adjust the number of instances used to host your model based on changes in your workload. 
- Model Compilation  SageMaker Neo automatically optimizes your models to run more effectively on cloud or edge hardware. 
- Elastic Inference (EI)  EI lets you add GPU-based accelerators to your hosting instances at a fraction of the cost of using a full GPU instance and supports any TensorFlow, MXNet, PyTorch, or ONNX model. 
- Inference Pipelines  This lets you deploy a linear sequence of two to fifteen containers that perform steps on your incoming data. Typically this sequence may involve a preprocessing step, a model prediction step, and a post-processing step done in real time. 
- SageMaker Model Registry  SageMaker Model Registry lets you catalog models for production, manage versions, add and manage manual approval steps for the model, and automate model deployment using continuous integration/continuous delivery (CI/CD).
- Managed Spot Training  You can use managed spot instances instead of on-demand instances to reduce the cost of training by up to 90 percent.
- To get predictions from trained models, you can either host a persistent endpoint for real-time predictions or use the SageMaker batch transform APIs to apply model predictions to an entire test dataset.
- For real-time predictions, SageMaker provides fully managed model hosting services and generates a private HTTPS endpoint where your model can return prediction outputs. You can deploy multiple production variants of the model to divert different percentages of traffic to different versions of your model. You can host multiple models and target these models from your client application calling the endpoint. And finally, after you deploy your model into production, you can use SageMaker’s Model Monitor to continuously monitor model quality metrics in real time and provide you with a notification when deviations such as data drift are detected. For batch predictions, SageMaker initializes the requested number of compute instances and distributes inference workload involving getting predictions for a large test dataset between these instances. Batch transform jobs create the same number of output files as input files, with an additional .out extension




- AWS DeepLens  The DeepLens ecosystem lets you learn about vision systems and deep learning by providing you with a fully programmable video camera and several pretrained models and examples. 
- AWS DeepRacer  The DeepRacer ecosystem lets you learn about reinforcement learning using a fully managed simulation and training environment, as well as a 1/18 scale RC (race car) car that can run your trained model. 
- AWS DeepComposer  DeepComposer is a fully programmable MIDI keyboard that lets you play, record, train, and generate music using generative adversarial networks (GANs). 
- AWS Panorama Device and SDK  This allows you to add computer vision–based applications to your IP camera setup. You can analyze video feeds from multiple cameras in parallel generating predictions from models that you trained and compiled on the cloud with SageMaker.


S3

- in the URL https://mybucket.s3.amazonaws.com/mykey/myobject .docx, mybucket is the bucket name, mykey/myobject.docx is the key, and myobject .docx is the object name.
- S3 Object Lock lets you implement write-once, read-many (WORM) policy. This lets you retain an object version for a specific period of time
- With S3 replication, you can copy objects to multiple locations automatically. These buckets can be in the same or different regions.
- S3 Select lets you query data without accessing any other analytics service using SQL statements. For more involved SQL queries, you typically use an analytics service like Amazon Athena or Redshift Spectrum to query data directly on S3.
- S3 storage is nonhierarchical. Although object keys may look like folder structures, they are just a way to organize your data.
- You can grant access to users using a combination of AWS Identity Access Management (IAM) and access control lists (ACLs).
- S3 supports both server-side encryption (SSE-KMS, SSE-C, SSE-S3) and client-side encryption
- stores objects using partitioning which is useful for quering later(partitioning can be handled by glue)
- Storage classes in s3:-
- s3 standard(general purpose) - 99.99 availability, used for frequently accessed data, low latency high throughput, used for big data analytics, mobile and gaming apps, etc.
- s3 standard-infrequent access(IA) - for data that is less frequently accessed, but requires rapid access when needed
- s3 one zone infrequent access 
- glacier (low cost mainly used for backup)
- s3 glacier instant retrieval - millisecond retrieval, great for data accessed over a quarter, min storgae of 90 days
- s3 glacier flexible retrieval - has 3 options: expediated(1-5 mins), std(3-5 hours) and bulk(5-12 hrs), min storgae of 90 days
- s3 glacier deep archive - std(12 hours), bulk(48 hours), min storgae of 180 days
- intelligent tiering - intelligently move objects after observing the pattern. no retrieval cost
- s3 security: 
- sse-s3 - encrypts s3 objs using keys handled and managed by aws
- sse-kms - key management service to manage encryption keys
- sse-c - when you want to manage your own encryption keys
- client side encryption



AWS EFS

- EFS is built for petabyte scale, and it grows and shrinks automatically and seamlessly as you add and remove data
- EFS also supports authentication, authorization, encryption at rest, and transit
- EFS filesystems can be mounted inside your VPC by creating a mount target in each availability zone so that all instances in the same availability zone can share the same mount target
- A group of data scientists working on projects together want to be able to easily share data and files, without first copying them into a notebook for preprocessing and then copying them back to S3 for training. They create a common EFS, mount the same EFS on multiple notebook instances, and use the EFS filesystem directly to train SageMaker models without the need to copy files back to S3


AWS FSx Lustre

- Amazon FSx is a fully managed, high-performance filesystem that can be used for large-scale machine learning jobs and high-performance computing (HPC) use cases
- Amazon FSx provides two types of filesystems: FSx for Windows and FSx for Lustre
- Lustre can support hundreds of petabytes of data storage and hundreds of gigabytes of aggregate throughput
- A customer performing large-scale distributed training has several terabytes of data on S3. Using FSx is a great fit in cases where you have terabytes of data (as with the autonomous vehicle datasets) and you need to train your model without redownloading all your data onto the SageMaker training instances.

- on AWS, your code is versioned using CodeCommit
- CodeCommit has a limit of 6 MB per file, so it may be impossible to version these large files. This is where data versioning tools like DVC (https://dvc.org) come to the rescue! DVC is used to track, version, back up, and restore snapshots of datasets by using familiar tools and AWS back-end storage services like S3 and EFS (which we discussed earlier). DVC uses local caches that can also be shared across users using services like EFS and can use S3 as a persistent store. 


AWS VPC

- VPC endpoints allow resources in your VPC to privately connect to supported AWS services and VPC endpoint services without using an Internet gateway or a NAT device; this enables more secure communication that limits all traffic to the private AWS network(which is powered by AWS PrivateLink).
- Some services, like Amazon Personalize and Amazon Forecast, do not support the S3 VPC gateway, and so pointing to training data in buckets that only allow VPC traffic will result in an AccessDenied error



AWS LAMBDA

- AWS Lambda is a SERVERLESS compute service that lets you run code without configuring any infrastructure. You can write these Lambda functions in Python, Node.js, Go, Java, C#, Ruby, PowerShell, or any custom runtime; you can also bring in your own containers built using Docker to run in Lambda. Lambda can be triggered by events coming in from API Gateway, SNS topics, S3 bucket changes, and DynamoDB streams, to name a few. Lambda scales automatically, from a few requests per day, to thousands of requests per second, which can be done in a concurrent (parallel) manner.
- Two types of permissions are commonly used with Lambda functions: execution roles and resource-based policies. Use execution roles to grant your Lambda function permissions to access AWS services and resources; use resource-based policies to give other accounts and AWS services permission to call your Lambda function.


AWS STEP FUNCTIONS

- AWS Step Functions is a serverless function orchestration service that lets you manage complex, distributed applications with built-in operational controls
- Step Functions lets you define your state machines with a JSON document using the Amazon states language, where each state can pass output data from the previous step to your own microservices, or AWS service integrations such as DynamoDB, SNS, Athena, Glue, EMR, or SageMaker, to name a few
- Let’s discuss some (not all) of the state types that you can include in your step functions workflow: 
- ■ Task state—Represents a unit of work done in a state; a task state can invoke a custom Lambda function with specific input parameters, or even call other supported AWS services. Note that custom activities can also run in EC2 instances, on ECS, or even on mobile devices. Tasks can also involve human approval steps, such as emailing links to approve. Services from the AWS Stack 
- ■ Choice state—Used to branch out based on some logic, similar to an if-then-else block, with ways to check most string and numeric logical operations (equals, greater than, not, etc.). 
- ■ Wait state—Used to delay the state machine from continuing for a specified number of seconds. 
- ■ Parallel state—Can be used to create parallel branches of execution in your state machine. The output of the parallel state is a list containing outputs from all branches. 
- ■ Map state—Used to iterate through multiple entries of an input array, where you can also process many tasks in parallel, defined by the MaxConcurrency value.




- The Cross Industry Standard Process for Data Mining (or CRISP-DM) can be used as a baseline to understand the various phases of the ML workflow
- Business problem ➔ ML problem framing ➔ Data collection ➔ Data exploration➔ Model training
- Structured data consists of data that has a well-defined schema and metadata needed to interpret the data such as the attributes and the data types. Tabular data is an example of structured data.
- Unstructured data is data that does not have a schema or any well-defined structural properties. Examples include images, videos, audio files, text documents, or application log files.
- Semi-structured data is data that does not have any precise schema such as data that can be in JSON format or XML data that you may have from a NoSQL database. You may need to parse this semi-structured data into structured data to make it useful for machine learning





AWS DATA REPOSITORIES 

- OLTP(online transaction processing) applications typically run on relational databases, and AWS offers a service called AWS RDS and aurora to build and manage this kind of data which is usually in row based format(oltp)
- For analytics and reporting workloads that are read heavy, consider a data warehouse solution like Amazon Redshift. Amazon Redshift uses columnar storage instead of row-level storage for fast retrieval of columns and is ideally suited for querying against very large datasets
- redshift is used for olap(online analytical processing)
- Amazon Redshift is now integrated with Amazon SageMaker via SageMaker Data Wrangler
- If your data is semi-structured, you should consider a NoSQL database like DynamoDB. DynamoDB stores data as key-value pairs and can be used to store data that does not have a specific schema.
- When you have data in diverse data repositories, you may want to centrally manage and govern the access controls to these datasets and audit that access over time. AWS Lake Formation is a data lake solution that helps you centrally catalog your data and establish fine-grained controls on who can access the data. Users can query the central catalog in Lake Formation and then run analytics or extract-transform-load (ETL) workstreams on the data using tools like Amazon Redshift or Amazon EMR.
- If your data is already on AWS, you can use AWS Data Pipeline to move the data from other data sources such as Redshift, DynamoDB, or RDS to S3. Data Pipeline uses a concept called activity types to perform certain action. An activity type is a pipeline component that tells Data Pipeline what job to perform.
- Data Pipeline has some prebuilt activity types that you can use, such as CopyActivity to copy data from one Amazon S3 location to another, RedshiftCopyActivity to copy data to and from Redshift tables, and SqlActivity to run a SQL query on a database and copy the output to S3.
- Data Pipeline also allows you to copy data out from S3 or DynamoDB to Redshift. Data Pipeline uses EC2 instances under the hood to migrate your data and can be run in an event-driven manner or on a schedule.
- If your data is in a relational format and you want to migrate data from one database to another, you can use AWS Database Migration Service (DMS)
- Understand the differences at a high level between DMS and Data Pipeline. Remember that Data Pipeline can be used with data warehouses such as Redshift and NoSQL databases such as DynamoDB, whereas DMS can only be used to migrate relational databases such as databases on EC2, AzureSQL, and Oracle. If your data is already in a relational database on AWS such as RDS and you want to migrate it to S3, choose AWS Data Pipeline instead of DMS
- AWS Glue is a managed ETL service that allows you to run serverless extract-transform-load workloads without worrying about provisioning compute. You can take data from different data sources, and use the Glue catalog to crawl the data to determine the underlying schema.
- Glue is a powerful service with capabilities such as data visualization using Glue Data Brew, serverless ETL, the ability to crawl and infer the schema of the data using data crawlers, and the ability to catalog your data into a data catalog using Glue Data Catalog. For the test, it is useful to know that you can use Glue to crawl and catalog data, convert data from one data format to another, run ETL jobs on the data, and land the data in another data source.



AWS KINESIS DATA STREAM

- Kinesis Data Streams is a service you can use to collect and process large streams of data records in real time. Once the data streams are in AWS, you can run a number of downstream applications such as real-time data analytics by sending the data to a dashboard.
- A data stream represents a group of data records, where a data record is a unit of data. Data records are distributed into shards. A shard represents a sequence of data records. Each shard supports 5 transactions/second for reads or up to 2 MB per second and 1,000 records/second up to 1 MB per second for writes. When you create a Kinesis Data Stream, you specify the number of shards. 
- ■ The retention period corresponds to the length of time the data is available in the stream. The default is 24 hours, but you can increase it to 365 days (8,760 hours). 
- ■ A producer puts records into the streams such as a web log server. You can use the Kinesis Producer Library (KPL) to build your producer application. 
- ■ A consumer gets records from the stream and processes them. Consumer applications often run on a fleet of EC2 instances. Kinesis Client Library abstracts a lot of the lowerlevel Kinesis data streams APIs to allow you to manage your consumer applications, distribute load across consumers, respond to instance failures, and checkpoint records.
- not serverless since you have to configure it with producer and consumer libraries
- you cannot write the data directly into s3 using kinesis data stream or data analytics, you need to use lambda or firehose as the intermediate thing





AWS KINESIS DATA FIREHOSE

- Often you may need to send streaming data directly to an end service such as Amazon S3 for storage, Redshift for querying, or Elastic Search/Splunk or some custom third-party endpoint such as Datadog. In this case, instead of writing complex producer-consumer applications like Kinesis Data Streams, you can simply use Kinesis Data Firehose
- Kinesis Data Firehose also allows you to use AWS Lambda functions to process the incoming data stream before it is delivered to the final destination service
- auto scaling 
- data conversions from csv/json to parquet/orc only for s3 though(through lambda)
- supports compression
- not real time(1 min latency/buffer interval)
- serverless data transformation with lambda



AWS KINESIS DATA ANALYTICS

- Kinesis Data Analytics lets you run SQL queries directly on streaming data
- Kinesis Data Analytics can be used to feed real-time dashboards or create real-time metrics and triggers for monitoring, notifications, and alarms.
- Kinesis Data Analytics does not source data from streaming sources directly, unlike Firehose or Data Streams. It requires either Data Firehose or Data Streams or S3 as an input, and sends the output of the SQL query back to either Kinesis Data Streams or Firehose for downstream processing to other consumers or AWS services.
- uses two ml based algos for anomaly detection: random cut forest, hotspots
- apache flink is used for streaming applications



AWS KINESIS VIDEO STREAMS

- producers: deeplens, etc consumers: rekognition, etc
- this service when you want to stream live video feeds into the AWS cloud
- Rekognition video can be used as a downstream “consumer” of a Kinesis Video stream.
- retain videos from 1hr to 10 years


AWS GROUND TRUTH 

- Amazon SageMaker Ground Truth is a service that you can use to label your image, text, audio, or even tabular data. SageMaker Ground Truth lets you outsource the labeling task to a public workforce (via Amazon Mechanical Turk) or a private workforce (either a thirdparty labeling company or your own private workforce within your organization) to label data. Ground Truth has built-in workflows for image and text labeling use cases such as image classification, object detection via bounding boxes, segmentation, text classification, entity recognition, video and audio labeling, or even 3D point cloud labeling. Furthermore, you can create custom workflows using Ground Truth if your use case doesn’t fit into these built-in types.
- Ground Truth also has an active learning capability where a machine learning model learns the labels provided by the human annotators. This is known as automated data labeling. The model then automatically labels images it has a high confidence in and only sends the low-confidence images to the human annotators for labeling. This can allow you to scale your labeling to a large number of images.


AWS EMR

- Amazon EMR is a fully managed Hadoop cluster ecosystem that runs on EC2. EMR allows you to choose from a menu of open-source tools, such as Spark for ETL and SparkML for machine learning, Presto for SQL queries, Flink for stream processing, Pig and Hive to analyze and query data, and Jupyter-style notebooks with Zeppelin. 
- Amazon EMR is useful when you want to run data processing and ETL jobs over petabytes of data. In addition to the Hadoop distributed filesystem for storage, EMR integrates directly with data in S3 using EMR File System (EMRFS).
 
AWS SAGEMAKER PROCESSING

- EMR gives you the scale to process petabyte-scale data on AWS, but sometimes your datasets are not as large or your data science team may not have the relevant Hadoop/big data experience that is required to run EMR jobs. For such use cases, you also have the option of using a tool like SageMaker Processing directly within SageMaker itself. SageMaker Processing lets you implement your processing logic using scikit-learn or PySpark,


A common design question is when to use SageMaker Processing versus EMR or Glue. SageMaker Processing is ideally suited for data scientists, and though it can process large amounts of data using, for example, PySpark, it is fully managed and the underlying compute runs in an AWS managed account. As of this writing, SageMaker Processing does not support spot instances or persistent clusters. EMR is ideally suited for the extremely large-scale (petabyte-scale) data requirements, but it does require familiarity with the Hadoop ecosystem. It runs on EC2 instances in your AWS account, and you can set up transient or persistent clusters depending on your workload. It is ideally suited for big data engineers


AWS GLUE

- Glue has the benefit over EMR in that it is serverless users have no need to manage the underlying compute or clusters to run their processing jobs
- In 2020, AWS released Glue Data Brew, which is a service to help data scientists visually inspect their data, explore the data, define transformations, and engineer features


AWS ATHENA REDSHIFT SPECTRUM

- Amazon Athena is a service that allows users to use SQL to access data directly on S3 without moving the data out of S3. it uses presto under the hood and it is serverless
- Redshift Spectrum is another analytics service similar to Athena that allows you to interactively query data on S3 without moving the data
- The difference is that Redshift Spectrum requires a running Redshift cluster and a connected SQL client, unlike Athena.
- Since Spectrum runs on Redshift, it has the advantage of Redshift-level scale and can run on exabytes of data using massive parallelism.
- If you are architecting for an analytics solution to interactively query data in S3 and already have a Redshift cluster up and running, consider Redshift Spectrum. Otherwise, consider using Athena instead.

Glue can be used for serverless Extract, Transform, and Load (ETL)—that is, where you don’t have to worry about provisioning compute. You simply author your data transformation and feature engineering code in Python, Scala, or PySpark and let Glue run the job and send the outputs to your S3 bucket. With EMR, you can spin up a cluster and submit Spark jobs to the cluster for large-scale processing using Spark.



One of the most common ways data leakage gets introduced in ML is during the normalization/standardization process. Always normalize/standardize after you have already split your data into train/test sets. Then you can normalize the training data and use those normalization scores (such as mean and standard deviation) to normalize the test set. If you normalize the entire dataset, and then split into train-test, leakage will often get introduced.

The most common forms of regularization are ridge (where you add an L2 penalty or quadratic penalty to the weights), lasso (where you add an L1 penalty or absolute value penalty to the weights), and elastic net (which combines the two). It is helpful to note that ridge regression tends to reduce the values of weights that are unimportant in predicting the labels, whereas lasso tends to shrink the weights to zero. If you have a linear regression problem with a lot of features, using lasso penalty as a start is a good solution to eliminate features that are unimportant. For this reason, lasso regression is also known as shrinkage. Note that ridge, lasso, and elastic net are all different forms of regression; they differ only in the loss function.

Remember, to avoid overfitting, you want to increase the minimum samples per leaf but decrease the maximum depth of the trees.

A popular question when deciding on an architecture may be when to use BlazingText versus document classification in Comprehend. Remember that you cannot deploy a Comprehend model outside of Amazon Comprehend since it is a fully managed service. BlazingText models, however, can be hosted as SageMaker endpoints or even deployed outside of SageMaker, so that may be a design consideration to be mindful of. (https://www.udemy.com/course/aws-machine-learning/learn/lecture/16609890#overview)

Another popular exam question is to make sure you understand the difference between random cut forest and random forest. The former is for unsupervised learning, whereas the latter is a common supervised learning algorithm

As a best practice, you will want to launch your training jobs not on your notebook or local machine but on a remote cluster, so the model can train in the background while you can continue exploring data, developing other algorithms in your notebook or IDE. To facilitate remote training, Amazon SageMaker offers a Python SDK (or you can also use the lower-level boto3 AWS Python SDK) to launch a remote training job. SageMaker will run your training job on remote EC2 instances that are hosted in an AWS account. As SageMaker is a managed service, these instances will be managed by SageMaker and not visible in your AWS account.

In 2020, AWS launched two new services that offer developers a low-code way to perform feature engineering. Glue announced Glue Data Brew, which is a visual data preparation tool. With Data Brew, you can visualize your data, clean your data, normalize your data, and use over 250 transformations to run common feature engineering transforms without authoring any code.  Since Glue Data Brew integrates with the Glue Data Catalog, it supports a number of data sources such as S3, Redshift, RDS, and Aurora and provides connectors for third-party data warehouses such as Snowflake. Also in 2020, SageMaker announced a service called Data Wrangler on SageMaker Studio, which pulls in data from S3 or Redshift, or by using Athena into SageMaker. Data Wrangler offers similar user interface (UI) capabilities and 300+ transforms, as well as the ability to author custom transforms with your own code. Data Wrangler also allows you to quickly build an ML model on a subset of the data with no code to get a baseline model.

just know that they are both low-code, UI-based feature engineering tools on AWS. In contrast, Glue and EMR both require you to write your own processing and feature engineering code

SageMaker notebook instances offer a capability for local mode training. Rather than storing all your code in notebook cells, with local mode, you can package your training code and all the associated dependencies into a Docker container. To build the Docker container, you can use one of the SageMaker built-in frameworks such as TensorFlow, MXNet, or PyTorch. Local mode does not work with SageMaker built-in algorithms, only with public SageMaker container images.

Local mode uses Docker compose and the NVIDIA Container Toolkit for Docker (https://github.com/NVIDIA/nvidia-docker) to first pull the public SageMaker training images or build a custom training container from scratch. Once the container is built locally on your SageMaker notebook instance, you can run your SageMaker training job as you normally would with the only change being the type of instance the training job is launched on. Instead of specifying a EC2 instance type, you would specify instance_type= 'local' or instance_type= 'local_gpu'. Note that local mode does not support distributed training, since the training runs on the notebook itself.

Note that local mode does not work for SageMaker built-in algorithms but allows you to use SageMaker’s managed frameworks for TensorFlow, PyTorch, or MXNet as well as bring your own custom container.

Once you are done training locally on a small dataset for development purposes, you will want to launch a training job on the entire dataset. As a best practice, you will want to launch your training jobs not on your notebook or local machine but on a remote cluster, so the model can train in the background while you can continue exploring data, developing other algorithms in your notebook or IDE. To facilitate remote training, Amazon SageMaker offers a Python SDK (or you can also use the lower-level boto3 AWS Python SDK) to launch a remote training job. SageMaker will run your training job on remote EC2 instances that are hosted in an AWS account. As SageMaker is a managed service, these instances will be managed by SageMaker and not visible in your AWS account.

To facilitate remote training, Amazon SageMaker offers a Python SDK (or you can also use the lower-level boto3 AWS Python SDK) to launch a remote training job. SageMaker will run your training job on remote EC2 instances that are hosted in an AWS account. As SageMaker is a managed service, these instances will be managed by SageMaker and not visible in your AWS account. The steps to launch a training job are as follows: 
■ Store your algorithm code and any associated dependencies in a local folder or in a GitHub repo. 
■ Pass in the training script as the entry point to SageMaker. This way, SageMaker knows to execute the training script once your code and the training data is loaded into the training container. 
■ Specify an IAM role. SageMaker training uses EC2 behind the scenes so you will need to pass SageMaker an IAM role along with IAM:PassRole permissions to make EC2 calls on your behalf. These EC2 instances do not live in your account, so you will not see them on the EC2 console. 
■ Specify the number of instances you want to launch. 
■ Specify the type of instance you want to launch. 
■ Specify the output S3 path where you want your model artifacts to be stored. 
■ Call the estimator.fit function by passing in the S3 path to your training data. You can also pass a validation dataset for testing purposes during training.


A single forward and backward pass performed using a mini-batch of data is called one iteration. One training cycle through your entire dataset is called an epoch.

When the training data does not entirely fit in the available memory, it is typical to either choose a larger instance with more memory, or perform data parallel training; again, first with multiple GPUs on a single instance, and then with multiple instances (with multiple GPUs each, in that order). In data parallel training, model replicas are created for each GPU (within an instance or across many instances as applicable), and training is usually controlled with one process per GPU. During each iteration of training, each process loads a mini-batch of data and passes this to a GPU, where gradient calculation is performed. In typical cases (called synchronous data parallel training), the gradients are collected from all GPUs and averaged before updating the model weights. The act of averaging these gradients is done by a worker called a parameter server.

Lastly, gradient averaging can also be done using the training workers, without the need for separate parameter servers. This is called the Ring-AllReduce approach, and is implemented in the popular Horovod Python package that can be used with most popular deep learning frameworks.

When the model itself is large and complex, such as in the case of many large vision- and language-based models, it may not fit in the memory of a single node. Another strategy in distributed training that helps with this is model parallel training. Here, the model itself is partitioned and distributed across many nodes. Then, an execution schedule passes batches of the data across multiple nodes in order for calculating gradients.

So when would you use data parallel training versus model parallel training? 

1. If your model can fit in a single instance’s memory, you should use data parallel training. 

2. If you still think your model doesn’t fit in a single instance’s memory, check again by gradually decreasing the size of the inputs or batch size and/or changing the model’s hyperparameters. 

3. If your model still doesn’t fit, try mixed-precision training (https://docs.nvidia .com/deeplearning/performance/mixed-precision-training/index.html). With mixed precision training, model parameters are represented using different numerical precisions so as to reduce memory consumption and also speed up training. Since reducing the mini-batch size can lead to a higher error rate, it may be beneficial to use mixed precision training to allow for larger mini-batch sizes. 

4. If your model still doesn’t fit, then use model parallel training.

CloudWatch allows you to monitor various processes, including training jobs using real-time logs, metrics, and events. You can also create custom dashboards and alarms that take some action when a specific metric reaches a threshold. cloudwatch alarms can monitor cloudwatch metrics to receive notifications when the metrics fall outside the specified levels. Amazon CloudTrail, on the other hand, records individual API calls made by, or on behalf of, entities or services in your AWS account. Lastly, Amazon EventBridge lets you respond to events specific to status changes in training jobs


￼
￼

If the dataset were truly balanced, then Accuracy would probably be the most appropriate metric to use

to reduce the amount of False Negatives, use Recall
to reduce the amount of False Positives, use Precision


DEPLOYMENT

- For AI services, the models are hosted in a managed AWS account that is owned by the service. If you need more control over the hosting container, architecture, security, and autoscaling groups, you can use Amazon SageMaker. In the case of Amazon SageMaker, you have access to the trained model artifact, but the model is hosted in an Amazon SageMaker owned account (i.e., not your AWS account). If you require even more control of your hosting infrastructure, you can use other services to deploy your trained model such as Amazon EC2, AWS Lambda, AWS Fargate, or Amazon EKS.
- Training a model on SageMaker results in a trained model artifact on Amazon S3 (a model.tar.gz file). To get predictions from this model, you can do one of the following: ■ Host a persistent endpoint for real-time predictions, where SageMaker provides fully managed model hosting services and generates a private HTTPS endpoint, where your model can return prediction outputs. ■ Use the SageMaker batch transform APIs to apply model predictions to an entire test dataset, where SageMaker initializes the requested number of compute instances and distributes inference workload involving getting predictions for a large test dataset between these instances.
- For realtime endpoints, SageMaker uses a trained model artifact (model.tar.gz file containing the saved model) from S3 and an inference container from ECR within multiple, auto-scalable instances (Figure 10.1). When a client needs a prediction from this model, a POST call (via one of the APIs) is made, which then is directed to one of the instances hosting the model. In the case of multimodel endpoints, each instance can hold multiple models.
- When hosting a real-time endpoint: 
- 
- ■ SageMaker starts serving a model using the following Docker command: docker run [Image] serve. 
- ■ SageMaker provides an HTTPS endpoint to access your model for predictions via authorized API calls. SageMaker handles authorization with IAM identity-based policies, where you can specify allowed and denied actions and resources, as well as condition keys. To learn more about this, please visit https://docs.aws.amazon .com/sagemaker/latest/dg/security_iam_service-with-iam.html. 
- ■ SageMaker performs health checks that time out in 2 seconds before hosting the endpoint; a failed endpoint health check will result in your model not being hosted. You can perform custom health checks, and it is routine to load a model successfully during a health check. This health check happens in the /ping:8080 route. 
- ■ Real-time invocations happen in the /invocations:8080 route and time out at 60 seconds; this means that the model must return a prediction within 60 seconds. It is routine for a model invocation to be much faster than this upper limit of 60 seconds. 
- ■ Removing a model artifact after a model is hosted may result in unpredictable results; although the endpoint may continue to provide predictions temporarily, model updates and autoscaling actions may fail. 
- ■ Outputs and errors recorded in the container or inference code are sent to CloudWatch logs. 
- ■ You can provide a source folder with your inference code and choose a prebuilt container rather than building and pushing a custom container. For more information on this, please see https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-frameworks-deep-learning.html. 
- ■ For more information on hosting, please visit the SageMaker documentation page on hosting: https://docs.aws.amazon.com/sagemaker/latest/dg/how-itworks-deployment.html.
- Autoscaling SageMaker endpoints, similar to autoscaling EC2 instances, adjusts the number of instances based on your incoming workload or prediction requests(or any metric that is available as a CloudWatch metric; more on this later). Autoscaling can increase or decrease the number of instances that hold your model. As such, there are two ways of setting up the autoscaling action: 
- ■ Target tracking scaling (recommended in most use cases): You define a scaling metric and a target value. Application Autoscaling will automatically create a CloudWatch alarm and calculate the adjustment required to serve your predictions in terms of number of instances. 
- ■ Step scaling (recommended for advanced users): Apart from defining the scaling metric and target, you also define how your endpoint should be scaled when the target threshold is crossed. To do this, you additionally define the lower bound, the upper bound, and the amount by which to scale



DEPLOYMENT(RELEASE STRATAGIES)

Assume that you have two models—model A and model B. Also assume that these models can be launched onto endpoint A and endpoint B

- Re-create Strategy Shut down endpoint A and use model B to launch endpoint B. This implies that there is some downtime where model requests are not being served; for this reason, re-create strategy may rarely be used in practice
- Ramped Strategy Gradually shift traffic from Production variant A to Production variant B behind the same endpoint (Figure 10.4). To do this on SageMaker, first create a new endpoint configuration and update the variants by passing in a list of new_production_variants
- Blue/Green Strategy When you only need to update the model behind an existing endpoint, use the ramped strategy, since you can point to a new model and gradually shift production traffic to a new variant. On the other hand, when you need to change the instance type and the model behind each production variant, create another production variant and then update the weight of the new variant to 100 after conducting tests. You can use the UpdateEndpointWeightsAndCapacities API call to switch the new variant. A blue/green deployment is typically for environment changes, not modeling

TESTING STRATAGIES

- Canary Testing Generally, canary testing involves targeting a small set of customers with the latest version of the application, obtaining early feedback, and incorporating it into the latest version before deploying it to the entire target audience through one of the deployment strategies above. On AWS, you can use Amazon CloudWatch Synthetics to create canaries, which are scripts meant to replicate or simulate user actions. This allows you to continue testing new versions of your model when there is no traffic to your endpoint, so you may discover early issues and bugs.
- Shadow Testing While you can use canary testing and simulations to discover some early issues, you may still not have metrics/feedback from a sufficient number of real users, since newer versions of your application are never made available to a significant portion of users. On AWS, you can route your incoming requests through a Lambda function, which then mirrors traffic from your production application to the shadow application
- A/B Testing


- the endpoint latency starts to increase as the number of incoming requests grows. This can result in a poor customer experience. To solve this, the engineers consider using SageMaker Autoscaling. Autoscaling dynamically adjusts the number of instances serving your model to meet your demand.

- Amazon Macie is a security service that uses machine learning to automatically discover, classify, and protect sensitive data in AWS. Amazon Macie recognizes sensitive data such as personally identifiable information (PII) or intellectual property and provides you with dashboards and alerts that give visibility into how this data is being accessed or moved. The fully managed service continuously monitors data access activity for anomalies and generates detailed alerts when it detects the risk of unauthorized access or inadvertent data leaks. Amazon Macie is available to protect data stored in Amazon S3
- Amazon SageMaker Neo enables training machine learning models once and running them anywhere in the cloud and at the edge. Neo optimizes models to run up to twice as fast, with less than a tenth of the memory footprint, with no loss inaccuracy
- Straight from the Amazon Neo Documentation page (link below) three advantages of using Amazon Neo with SageMaker models are: (i) run ML models with up to 2x better performance, (ii) reduce framework size by 10x, and (iii) run the same ML model on multiple hardware platforms.
- The Poisson is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space.
- Horovod is a distributed deep learning training framework that works with TensorFlow, Keras, PyTorch, and Apache MXNet. It is designed for speed and ease of use.
- AWS Step Functions lets you coordinate multiple AWS services into serverless workflows so you can build and update apps quickly. Using Step Functions, you can design and run workflows that stitch together services, such as AWS Lambda, AWS Fargate, and Amazon SageMaker, into feature-rich applications. Workflows are made up of a series of steps, with the output of one step acting as input into the next. They are perfect to run multiple ETL jobs that hand off the input to ML workflows. (The keyword in the question is ‘workflow’.)
- AWS IoT Greengrass then lets users build IoT solutions that connect different types of devices with the cloud and each other. Responding to local events in near real-time and operating offline when the Internet connection is not available are two advantages of using AWS IoT GreenGrass service.
- Athena operates the best on data in Parquet format.
- SPICE is a fast, optimized, in-memory calculation engine for Amazon QuickSight. SPICE is highly available and durable and could be scaled from ten to hundreds of thousands of users.
- This is an example where the dataset is imbalanced with fewer instances of positive class because of a fewer number of actual fraud records in the dataset. In such scenarios where we care more about the positive class, using PR AUC is a better choice, which is more sensitive to the improvements for the positive class.
- Chainer, sagemaker reinforcement learning, scikit learn and pytorch don't support network isolation
- By using Amazon Elastic Inference (EI), you can speed up the throughput and decrease the latency of getting real-time inferences from your deep learning models that are deployed as Amazon SageMaker hosted models, but at a fraction of the cost of using a GPU instance for your endpoint. Elastic Inference is supported in EI-enabled versions of TensorFlow, Apache MXNet, and PyTorch.
- AWS Panorama integrates with existing camera networks
- Sagemaker has describejob api with the failurereason option in case you want to check the reason for your training error
- sagemaker unsupervised algorithms that are parallelizable: random cut forest, pca, linear learner, image classification, knn
- sagemaker supervised algorithms that are not parallelizable: seq2seq, object2vec, semantic segmentation, kmeans, lda
- ipinsights algorithm only supports csv input
- seq2seq uses rnns and cnns under the hood
- blazingtext can be used as both supervised and unsupervised
- rds, s3, dynamodb have the ability to create snapshots
- recommended input format of images for sagemaker image classification: apache mxnet recordio
- seq2seq and factorization machines only support recordio-protobuf file type for training
- t, m, c, p, r instances type supported by aws sagemaker
- hbase is a non-relational distributed database that is not capable of doing machine learning tasks
- lambda is not suitable for long running processes like etl.. it has a processing time of 15 mins
- AWS managed SageMaker policies that can be attached to the users are AmazonSageMakerReadOnly, AmazonSageMakerFullAccess, AdministratorAccess, and DataScientist. DataEngineerAccess is not an AWS pre-defined SageMaker policy.


SageMaker has a feature called SageMaker Model Monitor to monitor any drift in data or model performance. Model monitoring works by first identifying a baseline on which the data is trained and deriving statistical distributions for the baseline. The SageMaker model endpoint is then instructed to capture and store a fraction of the incoming data traffic. The distribution of the incoming payload is then extracted and compared against the baseline, and any drift is output to CloudWatch as well as the SageMaker Studio UI. In addition to monitoring the model performance, continuous monitoring also involves logging the underlying infrastructure itself to ensure that it is healthy to meet your traffic, logging any unauthorized access of production endpoints for security reasons. API calls made to AWS services are logged in AWS CloudTrail, and Amazon CloudWatch can be used to monitor the compute performance of production endpoints for CPU usage, memory usage, and number of invocations. These CloudWatch logs can also be connected to thirdparty monitoring services such as Splunk or Datadog to visualize the endpoint performance in a centralized manner.

AWS recommends that you encrypt your data wherever possible and provides you with tools for doing so. Since most data for ML workloads on AWS is stored on S3, it is helpful to review the different data protection options you have on Amazon S3. You can encrypt your data client side, where the data is encrypted prior to being sent to AWS, or server side, where the data is encrypted at the destination. For server-side encryption of data using Amazon S3, you have the option of encrypting data with S3 managed keys, customer master keys (CMKs) stored in AWS KMS, or customer-provided keys that are managed by the customers. For client-side encryption, you can either use the AWS encryption SDK or use a CMK stored in AWS KMS to first encrypt the data prior to sending it to AWS. To learn more about encryption on S3, read the following document: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ UsingEncryption.html. In addition to encryption at rest, you should consider encryption in transit. All network traffic between AWS data centers is encrypted, as is traffic between VPCs peered across AWS regions. For the application layer, customers have a choice about whether to use an encryption protocol like Transport Layer Security (TLS).

Another key security consideration is to isolate your compute environments from the public Internet. By placing your resources in a VPC, you can launch resources in a virtual network dedicated to your account.

Note that GuardDuty uses machine learning to detect anomalies for threat intelligence. The other services we’ve discussed so far (CloudTrail, Config, CloudWatch) do not use ML but rather can log actions taken in your ML environments.

AWS Config is a service that provides a list of predefined and managed rules, and allows users to create custom rules using Lambda functions to perform detective and preventive controls on their AWS environments. These ensure that either any non-compliant resources are proactively terminated or admins are notified whenever resource state changes occur that put their systems out of compliance

Finally, if you are working in a regulated industry such as financial services or healthcare, you may need to ensure that the AWS services you use are in compliance with different regulations such as the Payments Card Industry (PCI) or Service Organization Controls (SOCs). AWS Artifact is a service that provides access to such compliance documentation. Remember the shared responsibility model here; customers are still required to ensure that any applications they build on top of AWS services meet the compliance and regulatory requirements they are subject to.

There are two ways to authenticate users into SageMaker. You can use either AWS IAM or AWS Single Sign-On (SSO). Although IAM groups and users are specific to an account, SSO allows you to manage users and access across all your AWS accounts in a centralized way. You can also use groups in SSO to assign permissions based on roles or job functions. Additionally, SSO integrates with many third-party authentication tools such as Okta, Ping Federate, Microsoft Active Directory (AD), and Azure AD. SSO authentication is only supported for SageMaker Studio. SageMaker Notebook instances support IAM-based authentication only.

Data protection consists of encrypting your data at rest and in transit. For encryption at rest, SageMaker notebooks, SageMaker training jobs, and processing jobs use AWS managed customer master keys (CMKs) by default to encrypt data at rest. However, you can specify customer managed CMKs if you wish to do so. For SageMaker training jobs, there are two kinds of CMKs you can use—a volume KMS key to encrypt the attached EBS volume, and an output KMS key to encrypt any model artifacts before they are stored in your output bucket. You need to specify these keys when calling the CreateTrainingJob API. The same controls apply to SageMaker processing jobs, SageMaker hyperparameter tuning jobs, and SageMaker AutoML jobs using AutoPilot. When serving a model using SageMaker endpoints, you can encrypt the EBS volume attached to the hosting instance.

As such, there are two ways of setting up the autoscaling action: Target tracking scaling (recommended in most use cases): You define a scaling metric and a target value. Application Autoscaling will automatically create a CloudWatch alarm and calculate the adjustment required to serve your predictions in terms of number of instances. Step scaling (recommended for advanced users): Apart from defining the scaling metric and target, you also define how your endpoint should be scaled when the target threshold is crossed. To do this, you additionally define the lower bound, the upper bound, and the amount by which to scale.

SSO authentication is only supported for SageMaker Studio. SageMaker Notebook instances support IAM-based authentication only.

For model artifacts that are designed for production use, using a tool such as AWS S3 Object Lock to prevent objects from being deleted (write-once-read-many [WORM]) is a recommended practice.

by separating test and production accounts, you can limit who can access those accounts and provide a security boundary, as well as ensure that if something fails, it will not affect other production systems and have limited impact or blast radius.

Like any software, a trained ML model should be recoverable in case of failure or loss. To prevent accidental deletions, using S3 Versioning or S3 Object Lock to store model artifacts intended for deployment is a good practice.

AWS DMS does not support DynamoDB as a data source

The feature_dim hyperparameter is a setting on the K-Means and K-Nearest Neighbors algorithms,
The sample_size hyperparameter is a setting on the K-Nearest Neighbors algorithm

The learning_rate hyperparameter governs how quickly the model adapts to new or changing data. Valid values range from 0.0 to 1.0. Setting this hyperparameter to a low value, such as 0.1, will make the model learn more slowly and be less sensitive to outliers.

The XGBoost algorithm is parallelizable and therefore can be deployed on multiple instances for distributed training. The inference container must accept POST requests to the /invocations endpoint.

From the Amazon SageMaker developer guide titled Amazon SageMaker Elastic Inference (EI) “By using Amazon Elastic Inference (EI), you can speed up the throughput and decrease the latency of getting real-time inferences from your deep learning models … You can also add an EI accelerator to an Amazon SageMaker notebook instance so that you can test and evaluate inference performance when you are building your models” Therefore, while you are in the development stage using jupyter notebooks, Elastic Inference allows you to gain insight into the production performance of your model once it is deployed.

For online testing you use live data. For offline testing you use historical data. When performing offline testing of your models, you deploy your trained models to alpha endpoints, not beta endpoints.

The Amazon Kinesis Data Streams Producer Library is not meant to be used for real-time processing of event data since, according to the AWS developer documentation “it can incur an additional processing delay of up to RecordMaxBufferedTime within the library”. The Amazon Kinesis Data Streams API PutRecords call is the best choice for processing in real-time since it sends its data synchronously and does not have the processing delay of the Producer Library. Therefore, it is better suited to real-time applications

Amazon Kinesis Data Analytics does not integrate directly with lambda

The Optimized Row Columnar (ORC) file format provides a highly efficient way to store Hive data

Data, ExplicitHashKey, PartitionKey, SequenceNumberForOrdering, and StreamName

 The Content-Type of text/csv without specifying a label_size is used when you have target data, usually in column one, since the default value for label_size is 1 meaning you have one target column.

While creating your model training job in the SageMaker console, you specify a regex pattern that is used for the metrics that your model training script writes to your logs. You can’t specify the metrics directly, you must use a regex pattern.

You can use the Amazon SageMaker model tracking capability to search key model attributes such as hyperparameter values, the algorithm used, and tags associated with your team’s models. This SageMaker capability allows you to manage your team’s experiments at the scale of up to thousands of model experiments.

Your inference pipeline is immutable. You change it by deploying a new one via the UpdateEndpoint API. SageMaker deploys the new inference pipeline, then switches incoming requests to the new one. SageMaker then deletes the resources associated with the old pipeline.

The AWS Glue FindMatches ML Transform uses machine learning capabilities to find matching records in your database, even when the records don’t have exactly matching fields. This type of matching is perfect for finding similar products in a products table. Setting the FindMatches ML Transform precision_recall parameter to precision is the correct parameter setting. You use this setting when you want to minimize false positives

The MAE is the correct regression metric to use when your dataset can be significantly influenced by outliers. Your dataset contains several outliers per region.

To receive inference requests, the container must have a web server listening on port 8080

aws batch https://www.udemy.com/course/aws-machine-learning/learn/lecture/16666348#overview 

best practises - read page 300

* Amazon S3: Object Storage for your data
* VPC Endpoint Gateway: Privately access your S3 bucket without going through the public internet
* Kinesis Data Streams: real-time data streams, need capacity planning, real-time applications
* Kinesis Data Firehose: near real-time data ingestion to S3, Redshift, ElasticSearch, Splunk
* Kinesis Data Analytics: SQL transformations on streaming data
* Kinesis Video Streams: real-time video feeds
* Glue Data Catalog & Crawlers: Metadata repositories for schemas and datasets in your account
* Glue ETL: ETL Jobs as Spark programs, run on a serverless Spark Cluster
* DynamoDB: NoSQL store
* Redshift: Data Warehousing for OLAP, SQL language
* Redshift Spectrum: Redshift on data in S3 (without the need to load it first in Redshift)
* RDS / Aurora: Relational Data Store for OLTP, SQL language
* ElasticSearch: index for your data, search capability, clickstream analytics
* ElastiCache: data cache technology
* Data Pipelines: Orchestration of ETL jobs between RDS, DynamoDB, S3. Runs on EC2 instances
* Batch: batch jobs run as Docker containers - not just for data, manages EC2 instances for you
* DMS: Database Migration Service, 1-to-1 CDC replication, no ETL
* Step Functions: Orchestration of workflows, audit, retry mechanisms



machine learning insights : anamoly detection, forecasting and auto-narratives

smote uses k-means

small batch size is a good solution for local minima problem


Docker containers are created using images, images are built from a docker file, images are saved in a repository > Elastic container registry(ECR)

everything is inside the /opt/ml container
- /opt/ml/input -> config/hyperparameters.json and resourceConfig.json 
- /opt/ml/data/ -> channel_name and input_data
- /opt/ml/model -> the files associated with the deployment code is inside(deployment container)
- /opt/ml/code -> training code goes to  and 
- /opt/ml/output -> output/errors 
- tensorflow doen't get distributed across multiple machiens automatically, so to do that there are two ways: use horovod or parameter server
- https://www.udemy.com/course/aws-machine-learning/learn/lecture/16643596#notes 
- spot instances can be interrupted, use checkpoints to s3 so that training can resume

for starting a training job, a sagemaker needs:
- url of the s3 bucket for training data
- ml compute resources(ec2 instance, etc.)
- url of the s3 bucket for output
- ecr path to the training code

For deploying:
- persisting endpoints for making individual predictions
- sagemaker batch transform to get predictions for an entire dataset

Linear learner can solve classification problems as well.
- it needs data in recordio protobuf format in float32 format or csv(first column as label)
- supports file or pipe mode
- data must be normalized(automatically does it for you) and data should be shuffled
- imp parameters: balance_multiclass_weights(gives equal imp tp each class in loss functions), learning_rate, mini_batch_size, L1, wd(weight decay/L2 reg)

XGBOOST

- takes in recordio protobuf, csv, libsvm, parquet as input format
- hyperparameters: subsample(to prevent overfitting), alpha, lambda, gamma, eta(all for regularization), eval_metric(loss metric), scale_pos_weight(helpful for adjusting unbalanced classes), max_depth
- supports gpu training since recent times(just need to set tree_method=gpu_hist)
- num_class and num_round are mandatory parameters
- only supports text/libsvm and text/csv

Seq2seq

- requires input in the form in recordio protobuf format(token must be integers, which is unusual as compared to other algos which require floats)
- must provide training data, valid data, vocabulary in tokanized files
- hyperparameters: batch_size, optimizer_type, learning_rate, num_layers_encoder, num_layers_decoder
- can optimize on accuracy, bleu_score(compare with other std translations) and preplexity(cross-entropy)

Object2Vec

- data must be tokenized and must be given to input as pairs
- contains two encoders: Enc1_network, Enc2_network - choose from hcnn, bilstm, pooled_embedding
- can use cpu or gpu and even multiple gpus
- set the inference_preferred_mode in the inference image to optimize for encoder embeddings rather than classification or regression
- uses vgg-16 or resnet-50(ssd) 

Image Classification needs MXnet recordIO format not protobuf(or raw jpg and png)
	
- uses resnet under the hood
- default image is a 3 channel 224x224 

Semantic segmentation

- built on mxnet gluon and gluoncv
- choice of algos: fully convolutional network, pyramid scene parsing, deeplabv3
- choice of backbones: resnet50/101 
- incremental training supported
- can only use gpu
 
Random cut forest is available to be used in kinesis analytics

Topic modelling is unsupervised

Automatic model tuning

- Dont optimize too many parameters at the same time
- limit your ranges to as small as possible
- use log scales when appropriate
- dont run too many training jobs concurrently

SAGEMAKER Debugger

- save internal state of the model at periodic intervals

In order to convert from json to parquet, use firehose, but in order to convert from csv to parquet, stream the data to s3 first then use glue

A MIME type is used so software ( like a browser for example ) can know how to handle the data.
If a server says "This data is of type text/csv" the client can understand that can render that data internally, while if the server says "This data is of type application/csv" the client knows that it needs to launch the application that is registered on the OS to open csv files.
text/csv is more generic.

If you plan to use gpus, make sure your container is nvidia-docker compatible. only the cuda toolkit should be included in the containers. dont bundle the nvidia drivers with the image

With the exception of InvokeEndpoint, CouldTrail is a service that provides a history of api calls.

￼


if the missing values are numeric, use knn, if the missing values are categorical, use neural net

LDA and neural topic modelling are unsupervised algos

comprehend can translate as well and lex can convert speech to text as well

Use deepar when there is no historical data for forecasting

If the model has a high specificity, it implies that all false positives (think of it as false alarms) have been weeded out. tn/(tn+fp)

emr, rds, elasticsearch require provisioning of servers

parallelizable: knn, linear learner, image classification, pca, random cut forest, deepar

non parallelizable: k-means, lda, semantic segmentation, seq2seq, object2vec

knn and xgboost are memory bound

seq2seq uses cnn and rnn under the hood

deeplense and step functions are serverless

You can select metrics and create graphs of the metric data using the CloudWatch console.

ecs(elastic container service) and eks(elastic kubernetes service) can be used to containarize aws ml in production

auc is scale invariant

ContentType	Algorithm																																																																																																		
application/x-image	Object Detection Algorithm, Semantic Segmentation																																																																																																		
application/x-recordio	Object Detection Algorithm																																																																																																		
application/x-recordio-protobuf	Factorization Machines, K-Means, k-NN, Latent Dirichlet Allocation, Linear Learner, NTM, PCA, RCF, Sequence-to-Sequence																																																																																																		
application/jsonlines	BlazingText, DeepAR																																																																																																		
image/jpeg	Object Detection Algorithm, Semantic Segmentation																																																																																																		
image/png	Object Detection Algorithm, Semantic Segmentation																																																																																																		
text/csv	IP Insights, K-Means, k-NN, Latent Dirichlet Allocation, Linear Learner, NTM, PCA, RCF, XGBoost																																																																																																		
text/libsvm	XGBoost																																																																																																		
apache mxnet recordio()	sagemaker image classification																																																																																																		

AWS lambda has an execution limit of 15 mins

Preventative controls encompass the AWS CAF Security Perspective capabilities of IAM, infrastructure security, and data protection. They are (i) IAM, (ii) Infrastructure security, and (iii) Data protection (encryption and tokenization). Detective controls detect changes to configurations or undesirable actions and enable timely incident response. They are (i) Detection of unauthorized traffic, (ii) Configuration drift, (iii) Fine-grained audits.

On emr by default, each cluster writes log files on the master node. These are written to the /mnt/var/log/ directory.

9980016303

Amazon SageMaker Build-in algorithms can stream dataset directly from S3 using Pipe Mode to users' training instances instead of data being downloaded first. This mode is unlike File mode, which downloads data to the local Amazon Elastic Block Store (EBS) volume before starting the training.

￼


normalize the data if the distribution is not gaussian/normal, if different features are on different scales
standardize the data if it has a lot of outliers

check all sagemaker iam policies
check amazon defense, sabre, mace, neo, cloudformation, splunk, relay, proposal, choice 



Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
A better approach to standardizing input variables in the presence of outliers is to ignore the outliers from the calculation of the mean and standard deviation, then use the calculated values to scale the variable.
This is called robust standardization or robust data scaling.
Logarithm transformation and Robust Standardization are the correct techniques to address the outliers in data.

Normalization - Normalization scales all values in a fixed range between 0 and 1. This transformation does not change the distribution of the feature and due to the decreased standard deviations, the effects of the outliers increases. So this option is incorrect.

ML speciality aws certification

1

1. You have raw text data stored in S3 and would like to use each document to train a custom text classification model. What is the easiest way to achieve this? 

A. Download all your data and work with an open-source framework on your laptop. 
B. Use Comprehend Custom labels to train a custom document classification model. 
C. First use SageMaker Processing to preprocess your data; then use the SageMaker built-in Blazing text algorithm to train and deploy your model. 
D. None of the options is correct. 

2. A customer would like to run computer vision models at a manufacturing facility and already uses IP cameras and custom edge devices for other purposes. The customer is a current user of SageMaker and needs suggestions on how to deploy these models. Which of the following options would you as a solutions architect suggest? 
 
A. Replace all IP cameras with DeepLens cameras, and use SageMaker models at the edge. 
B. Use outposts and attach cameras directly to Outpost. 
C. Purchase “smart cameras” from a vendor and retrain your models on the vendorprovided software. 
D. Download and use SageMaker trained models on the custom edge devices. 
 
3. A marketing data provider has 50 GB of time series data from various customers and would like to train a forecasting model to predict future sales. The customer uses an open-source algorithm on premises and is exploring ways to build multiple forecasting models based on cohorts of customers. Which of these solutions will work for this company? 

A. Use Amazon Forecast. It automatically recognizes cohorts and can easily handle up to 100 GB of files on premises or on S3. 
B. Use the open-source algorithm on SageMaker either by using Script mode or by bringing in a custom container and pointing the training job to data on S3. 
C. Redshift is the best option to both store and query data. It can also be used to forecast data in this case. 
D. None of the options is correct. 

4. A customer using SageMaker Studio has been manually running each step in a complex workflow. What is the easiest way to automate and manage these manual steps? 

A. Use SageMaker Pipelines. It is integrated with Studio, and converting the manual steps to a workflow is easy with the Python SDK. 
B. Move all steps to Step functions. Author the individual steps on Studio, but run pipelines in the Step functions. 
C. Move all steps to Managed Workflows for Apache Airflow. Author the individual steps on Studio, but run pipelines in Airflow. 
D. Move all steps to an EC2 instance, and use a Bash script to run each step in succession.

5. A customer currently uses Spark on premises to transform datasets for machine learning purposes. The customer is new to AWS and is aware of training options that are available on SageMaker. The customer would like to reuse Spark code that they have developed as is but make it part of their machine learning lifecycle on AWS. What solution will require the least amount of maintenance and would integrate well with other steps in the machine learning lifecycle? 

A. Use EMR to run on-demand Spark jobs. 
B. Use the Spark processing container provided by SageMaker and prepare data for training steps that will also use SageMaker. 
C. Use Glue DataBrew to import your Spark code and run as part of a data preparation pipeline. 
D. Set up an EC2 instance that replicates the on-premises setup. Since the setup on AWS now matches the on-premises setup, the customer can easily run Spark jobs without any additional effort. 

6. A customer running a streaming service has 10,000 audio files in S3. The customer would like to easily label these audio files and use them in a deep learning algorithm for music genre classification. Which solution will allow the customer to achieve this? 

A. Use a built-in UI template for audio classification on SageMaker GroundTruth, followed by a built-in audio classification algorithm to train the model. 
B. Use a built-in UI template for audio classification on SageMaker GroundTruth, followed by a custom audio classification algorithm to train the model. 
C. Use a custom UI template for audio classification on SageMaker GroundTruth, followed by a built-in audio classification algorithm to train the model. 
D. Use a custom UI template for audio classification on SageMaker GroundTruth, followed by a custom audio classification algorithm to train the model. 

7. A media company wants to process image data to detect persons, objects, and text from a database of images, but the company is concerned about their lack of machine learning expertise to build and deploy a custom solution. Which AWS service would you advise them to use to solve this problem? 

A. Amazon Comprehend 
B. Amazon Rekognition 
C. Amazon SageMaker 
D. Amazon Textract 

8. An asset management firm would like to build a chatbot-based solution to automate advice given to their clients by their financial advisors. They are concerned that due to their diverse global client base, the chatbot will need to translate incoming text into English before the advice can be rendered. What services would you use to build this solution? 

A. Lex, Translate 
B. Lex, Polly 
C. Translate, Polly 
D. SageMaker, Lex

9. A retail company wants to build a forecasting model to forecast demand for their products.
They have thousands of products and related product metadata. Although they have tried a
few models like ARIMA on premises, they are concerned with the model performance and
are also looking for a solution that can be scaled and deployed easily. Another concern they
have is that they do not want to understock their warehouses. What solution would you
recommend?

A. Train DeepAR on Amazon SageMaker for scalability and pick MAPE loss to solve the
understocking problem.
B. Train ARIMA on EC2 and use EC2 AutoScaling to solve the scalability issue.
C. Use Amazon ETS on Amazon Forecast. Include product information as item metadata.
Pick a 0.75 weighted quantile loss metric to solve the understocking problem.
D. Use DeepAR+ on Amazon Forecast. Include product information as item metadata. Pick
a 0.75 weighted quantile loss metric to solve the understocking problem.

10. You are trying to get your organization excited about machine learning. You host a tournament where employees can race a car around a race track that is programmed using reinforcement learning to teach them about applications of ML to real-world scenarios. Which AWS service is suited for this activity?

A. AWS Deep Lens
B. AWS Deep Composer
C. Amazon S3
D. AWS DeepRacer

11. You have trained an ARIMA-based forecasting model to forecast electricity prices in ZIP codes across the country. You want to use a metric that penalizes the model differently for under-versus overpredicting the price. Which metric would you use?

A. Weighted quantile error
B. Root mean squared error (RMSE)
C. Mean squared error (MSE)
D. Mean absolute percentage error (MAPE)

12. You want to train a single model across a multitude of time series ranging in the thousands.
You also have contextual data associated with the time series as a related time series, but the
related time series data does not extend in the prediction interval. Finally, you wish to use a
fully managed service to produce the ML model instead of developing your own algorithm
code from scratch. What service and algorithm would you use?

A. Amazon Personalize, multi-arm bandits
B. Amazon SageMaker, XGBoost
C. Amazon Forecast, CNN-QR
D. Amazon Forecast, DeepAR

13. A major sports company wants to detect helmets on players to ensure player safety. The
company has terabytes of video, but it is largely unlabeled. What AWS service would you use
to label the data?

A. Amazon Comprehend
B. Amazon SageMaker Ground Truth
C. Amazon SageMaker Processing
D. Amazon Forecast

14. Consider the same use case as in the previous two questions. Having trained the object detection algorithm, you want to deploy it in production. However, the incoming raw video first needs to be processed before it can be sent to the model for inference. This processing code is written in Spark. You want to jointly deploy the Spark-based processing code and the inference code. Which AWS tool lets you do this?

A. Inferentia
B. Neuron SDK
C. Inference Pipelines
D. SageMaker Model Monitor

15. You work for an insurance firm trying to automate insurance claims processing. As a first
step, you want to parse PDF documents and extract relevant entities. What AWS service
could you use to get started with entity detection without much ML experience?

A. AWS SageMaker
B. Amazon Comprehend
C. Amazon Kendra
D. Amazon Personalize

16. You are the head of a law firm trying to modernize your internal document search systems.
What AWS service would you use where users can type their questions and the service will
parse the question and provide the most relevant collection of documents that may match
the response?

A. Amazon Kendra
B. Amazon Comprehend
C. Amazon Forecast
D. Amazon Rekognition

17. You work for an insurance firm trying to automate insurance claims processing. As a first
step, you want to perform optical character recognition (OCR) and extract forms and tables from PDF documents. What AWS service could you use to get started with this use case
without having to build your own or use an open-source OCR solution?

A. AWS SageMaker
B. Amazon Textract
C. Amazon Kendra
D. Amazon Comprehend

18. You have some custom PySpark code that you use to process data prior to training an ML
model on that processed data. Which of the following AWS tools can be used to process the
data? (Choose all that apply.)

A. SageMaker Clarify
B. AWS Glue
C. SageMaker Processing
D. Amazon TimeStream

19. Which AWS services allow you to build ML Ops pipelines by defining a directed acyclic
graph (DAG) that can be executed to process data, train a model, and deploy the model?

(Select all that apply.)
A. AWS Step Functions
B. AWS SageMaker Pipelines
C. Amazon CodeCommit
D. Amazon CodeBuild

20. Which AWS service proactively detects bottlenecks and defects in your code and offers
suggestions to improve based on AWS code best practices for code in AWS CodeCommit
or GitHub?

A. AWS Guru Code
B. AWS Code Guru
C. AWS DevOps Guru
D. AWS Lookout for Code


=========================================================================
2
=========================================================================


1. A customer who is familiar with Lambda is curious to try training machine learning models on Lambda. The customer says that the data is usually about 100 MB in size, and the generated models are usually less than 10 KB. What will you, as a solutions architect, suggest as next steps for the customer?

A. Tell the customer that Lambda cannot run machine learning workloads and tell her that she may be thinking of SageMaker when she mentioned Lambda.
B. Tell the customer that though she can use Lambda for this purpose, the 100 MB dataset may be too large for Lambda to handle.
C. Tell the customer to explore using a custom container for Lambda that includes the machine learning framework of choice, and read data from S3, and write trained models back to S3.
D. Tell the customer she can use SageMaker APIs to directly run training on Lambda. SageMaker manages the containers for her, and all she has to do is submit a script containing training code.

2. An ML engineer is trying to figure out a way to connect an EC2 instance that runs a business-critical application to Kendra that contains a trained index with data from some internal websites. The EC2 instance is in a VPC and cannot query the Kendra index. Which solution will enable querying the Kendra index from this EC2 instance?

A. Since Kendra is a managed service, you cannot access it from your own EC2 instance in
a VPC.
B. Since Kendra is a managed service, you can contact AWS support to place it in your VPC
so that you can securely access it.
C. Since Kendra is a managed service, you can establish a private connection between your
VPC and Kendra by creating an interface VPC endpoint and continue to use Kendra APIs.
D. Since Kendra is a managed service, you can establish a private connection between your
VPC and Kendra by creating a Gateway VPC endpoint and continue to use Kendra APIs.

3. A customer is using Step Functions to orchestrate batch transform workloads on Amazon SageMaker. The customer wants to start multiple batch transforms at the same time. What type of state should the customer use?

A. Parallel state
B. Map state
C. Choice state
D. Task state


4. A company that builds an intelligent search service would like to first call Amazon Textract and then use Amazon Comprehend for each paragraph found as raw text from Textract. What services can be used in this architecture for an end-to-end serverless implementation, assuming that the input files can be stored in S3?

A. S3
B. S3 and Lambda
C. S3 and EC2 instances
D. S3, Lambda, Step Functions, and DynamoDB

5. Your customer is interested in exploring reinforcement learning for building indoor navigation systems for their fleet of workshop robots. What services on AWS can help them with their product?

A. AWS Lambda and EC2 instances
B. Amazon Personalize and DynamoDB
C. Amazon SageMaker RL and RoboMaker
D. Deep Graph Library and Neptune

6. You want to train ML models on terabytes of data using SageMaker but are concerned with the time it takes to load such massive datasets into the SageMaker training instance attached storage. What service could you use instead?

A. Use separate Elastic Block store volumes
B. Use Amazon S3
C. Use FSx for Lustre
D. Use Redshift

7. You are building a Step Functions workflow to compare the outputs of your ML model inference to Ground Truth data. You want to add branching logic that forks the workflow based on the results. Which state would you use?

A. Task state
B. Parallel state
C. Choice state
D. Pass state

8. What networking construct would you use to ensure that AWS services only access your data in Amazon S3 using AWS PrivateLink?

A. Security groups
B. VPC endpoints
C. AWS Transit Gateway
D. NAT gateway


9. What resource-based policy can you use to restrict which AWS services can access your S3 buckets?

A. IAM role
B. IAM policy
C. S3 bucket policy
D. Service control policy

*** 10. You are an MLOps engineer working to deploy ML models built by your data science teams. There is a considerable amount of code and dependencies that can be reused across these models such as ML frameworks, as well as custom libraries developed by scientists. The models are all relatively small and can be deployed using Lambda. What feature of AWS Lambda would you use to promote code reuse and package code into zip files that can be shared across multiple Lambda functions?

A. Lambda runtime
B. Lambda layer
C. Lambda provisioned concurrency
D. Lambda function

*** 11. You are an MLOps engineer working to deploy ML models built by your data science teams. The models can be deployed using Lambda functions but are subject to a low-latency serving requirement. You are concerned that the cold start problem due to initializing the execution environment will add additional latency during serving. What can you use to mitigate this concern?

A. Use Amazon SageMaker for inference instead.
B. Use Lambda provisioned concurrency.
C. Use EC2 to host your models.
D. Lambda functions don’t suffer from cold start.

12. You have built a Step Functions workflow to retrain your ML models whenever a sufficient amount of new data is stored in your S3 bucket. You want to now automate this to trigger the pipeline in an event-driven manner. What AWS service would you use to start the execution of the Lambda function when new data is added?

A. AWS EventBridge
B. Nothing; Step Functions can automatically be triggered by new data in S3
C. AWS Lambda
D. AWS CodeCommit

13. You have designed an EKS cluster for training large-scale transformer models of hundreds of millions of parameters for NLP applications. You wish to attach a filesystem to the cluster that allows you to store public datasets used for training, algorithm logs, and so forth. Which service would you use?

A. AWS FSx for Lustre
B. Amazon EBS
C. Amazon S3
D. Amazon Lake Formation

14. A customer would like to use AWS Lambda to do large-scale video processing in a serverless fashion. The processed frames will be used to train downstream computer vision models. However, processing such a large dataset at scale requires several Lambda functions and custom code that cannot fit in the size of a Lambda layer. What storage service would you recommend instead to store the custom code and share video data across all the Lambda functions?

A. Amazon EBS
B. Amazon EFS
C. Amazon S3
D. Amazon RDS


=========================================================================
3
=========================================================================


1. The CEO of an organization recently attended a conference and would like to embark on an
ML journey. While the CTO has a few data scientists working in her team, the CEO is unsure
what the first step is to solve a problem using machine learning. As the chief architect under
the CTO, what would you recommend to the CEO?

A. Identify the data and check the data quality.
B. Tell the CEO that ML projects rarely end up in production so it is not worth the time
spent.
C. Ask your data scientists to train a few models on public datasets and present the outcomes to the CEO.
D. Ask the CEO to nominate a few lines of business (LoBs) that you can work with to identify a business problem that can be solved using ML.

2. Which of the following are the correct steps in the CRISP-DM methodology for the ML lifecycle? (Not all steps are included.)

A. Business problem ➔ ML problem framing ➔ Data collection ➔ Data exploration➔ Model
training
B. Business problem ➔ ML problem framing ➔ Data collection ➔ Model training ➔ Data
exploration
C. Business problem ➔ ML problem framing ➔ Model training ➔ Model evaluation
D. Data collection ➔ ML problem framing ➔ Business problem ➔ Model training

3. A business stakeholder comes to you with a business problem to extract a fixed set of known
text from documents if they match exactly. He asks you if this problem can be solved using
machine learning. As a data scientist, what advice would you give?

A. Yes, you can solve this using image recognition.
B. Yes, this is a classification problem in machine learning.
C. This problem can be solved with simple regular expression (regex) matching, so it is not
an ML problem.
D. Yes, treat this as a natural language processing problem and solve it using named entity
recognition (NER).


4. A business stakeholder comes to you with a business problem to predict anomalies in sales data. The business has a small set of rules to detect anomalies today but also has historical data of labeled sales anomalies going back a few years. However, the stakeholder is concerned that as the data size grows, the business rules will be insufficient and become cumbersome to maintain. How would you work with the business to implement the least complex solution? (Choose all that apply.)

A. Develop a rule-based model initially that incorporates the business rules outlined by the
stakeholders. As the data size grows, scale out the server to detect anomalies in parallel.
B. Identify whether the historical sales data can be used to train a random cut forest
anomaly detection model using Amazon SageMaker.
C. Develop a rule-based model initially that incorporates the business rules outlined by the
stakeholders. If the data size grows, simply scale up the server running the business rules.
D. Since the rule-based approach is not scalable on its own, train a reinforcement learning
model to learn the rules based on historical sales data.

5. Which of the following are the correct steps in the CRISP-DM methodology for the ML lifecycle? (Not all steps are included.)
A. Business problem ➔ ML problem framing ➔ Data collection ➔ Data exploration➔ Model evaluation
B. Business problem ➔ Model hosting ➔ Data collection ➔ Model training ➔ Data exploration
C. Business problem ➔ ML problem framing ➔ Model training ➔ Model evaluation
D. Business problem ➔ ML problem framing ➔ Data collection ➔ Data exploration➔ Model training

6. Your CEO wants your organization to start investing in ML initiatives that are relevant to
the business. The CEO is frustrated that most ML projects in the company are still using
public datasets and not relying on first-party data and therefore are not insightful. You
have worked with your AWS account teams and hosted a discovery workshop to identify
a few strategic initiatives that are important for the business. What is a possible next step
you can take?

A. Of the use cases you have found, identify which ones can be solved with ML. Then
quickly determine if you have the datasets available in order to train ML models. Then
conduct a proof of concept on one of the use cases to quickly demonstrate value to your
CEO.
B. Pick any one use case of the ones you have identified, and train an ML model using
public data relevant to that use case.
C. Of the use cases you have found, identify which ones are ML problems. Then look for
available public datasets from websites such as Kaggle.com that are closely relevant to
your use case. Train an ML model and showcase the results to your CEO.
D. Of the use cases you have found, identify which ones can be solved with ML. Then
quickly determine if you have the datasets available in order to train ML models. For
ones that do not have clean, labeled data available, start by labeling data.


7. A business stakeholder approaches their AWS Solutions Architecture team about predicting product defects in assembly lines using computer vision. What is the first question you should ask this stakeholder?

A. You should tell them that this is not an ML problem and will require humans to manually determine whether products are defective or not.
B. You should ask them if they have high-quality labeled images available that label defective and nondefective products.
C. While defect detection is an ML problem, it is not a computer vision problem and can be
solved using a simple statistical algorithm like ARIMA.
D. Defect prediction does not require labeled data and can be accomplished using unsupervised learning algorithms such as principal component analysis.

8. Consider the same use case as in the previous question. The stakeholder goes back and reports that while they have some labeled data available, it is certainly not in the hundreds of thousands to millions of images. They have read that training deep learning algorithms requires a lot of data and are concerned that it will take a lot of time and money to source this data. What advice would you give the stakeholder? (Choose all that apply.)

A. They should consider using pretrained computer vision (CV) models available from open
source and apply transfer learning on the small labeled dataset to begin with.
B. They should abandon the project because without millions of images, the ML model will
not perform well.
C. They can consider a service like SageMaker Ground Truth Automated Labeling to automatically label images using ML as a way to scale the number of labeled images.
D. They should train a CV model from scratch on the small dataset they have.


=========================================================================
4
=========================================================================


1. A business stakeholder has approached you about creating a model to recommend items to
users. However, although she has data about the users themselves, she does not have any
labels. As a data scientist, which of the following options would you recommend to her to get
started with most quickly?

A. All machine learning problems require labels; since she does not have labels, this is not
an ML use case.
B. All ML problems require labels, so she should spend the next 6 months first getting
labeled data users have purchased before building an ML model.
C. Start simple by building a clustering algorithm that clusters the users into groups.
Recommend items to new users based on the items purchased by users in their cluster.
D. Train a classification model to predict the item for each user.

2. Which of the following are the correct steps in the CRISP-DM methodology for the ML lifecycle? (Not all steps are included.)

A. Business problem ➔ Data collection ➔ Data exploration➔ Model training ➔ ML problem
framing ➔ Model evaluation
B. Business problem ➔ ML problem framing ➔ Data collection ➔ Data exploration➔ Model
training ➔ Model evaluation
C. Business problem ➔ Data collection ➔ Data exploration ➔ Model training ➔ Model evaluation ➔ ML problem framing
D. Data collection ➔ ML problem framing ➔ Business problem ➔ Model training

3. A business stakeholder comes to you with a business problem to extract entities from documents. The documents are quite specific to your business and the entities have custom business verbiage that is not found in common parlance. Currently, the business stakeholder does not have much labeled data available. What advice would you give him to proceed with this use case?

A. Entity recognition is an ML problem that doesn’t require labeled data. You can use a
clustering algorithm to discover entities.
B. Entity recognition requires labels. Since the relevant entities correspond to verbiage that
is not commonly found, you will need to train a custom model to detect them. For this,
you will need to first develop a strategy to acquire labels. Advise the business stakeholder that you will need to factor in data labeling as part of this project.
C. Entity recognition requires labels. Simply pick up an off-the-shelf entity recognition
model that is trained on the Wikipedia text corpus for detecting the relevant entities.
D. Entity recognition is not an ML problem. Advise him to write a set of rules to detect
entities.


4. A business stakeholder comes to you with a business problem to extract entities from documents. The documents are quite specific to your business and the entities have custom business verbiage that is not found in common parlance. You have determined that the documents are currently in PDF format. What follow-up question would you ask the stakeholder?

A. Nothing; simply tell them that Textract can process PDF documents and point them to
the Amazon Textract documentation.
B. Entity recognition requires labels. Since the relevant entities correspond to verbiage that
is not commonly found, you will need to train a custom model to detect them. For this,
you will have to first develop a strategy to acquire labels. Advise the business stakeholder that you will need to factor in data labeling as part of this project.
C. Entity recognition requires labels. Remind the customer that there are several off-theshelf entity recognition models that are trained on the Wikipedia text corpus for detecting the relevant entities.
D. Ask the customer whether the PDF documents are in Amazon S3.

5. A business stakeholder comes to you with a problem to generate a know your customer (KYC) score for business-to-business (B2B) transactions. The stakeholder has public data about their customer, history of past transactions, dates, transaction amounts, and whether they were approved or denied. Today the KYC modeling is rules-based, but the stakeholder would like to augment this with machine learning to reduce false positives. What advice would you give the stakeholder? (Choose all that apply.)

A. Tell the customer that a rules-based approach is likely to outperform ML since this is
not an ML problem.
B. Ask the customer whether they have considered stitching the transaction data with the
Ground Truth data available about the quality of the customers to generate a labeled
dataset.
C. Consider a clustering approach to cluster customers based on the transaction data.
D. Probe the customer on why they are considering augmenting the rules-based approach
and what pain points they have with their current approach.

6. A business stakeholder comes to you with a problem to generate customer profiles and similarity scores to better understand their customers’ purchasing behavior. The goal is to use this to upsell and cross-sell products to them based on items that related customers may have purchased. The stakeholder is not an expert in ML and is reaching out for advice on where to start. In particular, they are concerned about the cost of generating labeled data. However, they do have prior customer purchase data available. How would you help the stakeholder address this business problem?

A. Tell them that customer segmentation is a supervised ML problem and therefore they
will need to generate labeled data.
B. Tell your customer that customer segmentation is a good place to start, and it can be
addressed using unsupervised machine learning methods, so labeled data is not necessary
to begin with.
C. Tell your customer that Amazon Personalize is a service that lets you recommend items
to customers based on prior purchase behavior and they should use that service.
D. Tell your customer that Amazon Lookout for Metrics is a service that lets you recommend items to customers based on prior purchase behavior and requires no prior ML
knowledge.

7. Consider the same use case as in the previous question. Having determined the kind of
approach to take, the customer is satisfied that there is a path forward. However, it is unclear
how the outputs of the model will impact the business. What questions might you ask to
glean additional data in this regard?

A. Ask them how the customer segments determined by the ML model are related to the
key customer segments that are identified by the business.
B. Ask them how the clusters produced by the ML model will be used to determine the
cross-sell/upsell strategy.
C. Ask them how they define success and how they plan to measure and monitor whether
the customers are actually purchasing the products recommended by the strategy.
D. All of the above options are correct.

=========================================================================
5
=========================================================================


1. You are building an ML solution to identify anomalies in sensor readings from a factory. The sensor publishes numerical data every second. What kind of data is sensor data?

A. Structured, time series
B. Unstructured, time series
C. Image data
D. Text data

2. You are building an ML solution to identify the sentiment of news and social media feeds about a company. What kind of data is the incoming news data and what kind of ingestion use case is it?

A. Structured, CSV data, streaming data collection
B. Semi-structured, JSON, streaming data collection
C. Unstructured, text corpus, batch data collection
D. Unstructured, text corpus, streaming data collection

3. Refer to Table 5.1 for this question. If you are trying to predict the housing price, the column “Price (in USD)” represents what ML data concept?

A. Label
B. Categorical feature
C. Data point
D. Continuous feature

4. Refer to Table 5.1 for this question. If you are trying to predict the housing price, the column “ZIP code” represents what ML data concept?

A. Label
B. Categorical feature
C. Data point
D. Continuous feature

5. As the head of AI/ML for a large enterprise, you are building an ML platform. Since the platform will be used by different lines of business (LoBs), you are creating separate S3 buckets for data scientists in those LoBs in different AWS accounts. Your IT security team has asked you to ensure that the data stored in those LoB buckets has fine-grained access controls and entitlements established at the column level and is cataloged in a central repository. What AWS service would you use to achieve this?

A. AWS IAM to establish access controls and AWS Glue Catalog to catalog data
B. Store the data catalog as a table in Amazon Redshift
C. Hive Catalog
D. AWS Lake Formation

*** 6. You have a large amount of data stored in Redshift in Parquet format that you need to convert to CSV before storing in S3. What is the simplest data ingestion solution to sanitize this data into the proper format for machine learning?

A. Use AWS Data Pipeline using a CopyActivity to copy data directly from Redshift to S3.
B. Use AWS Glue to move the data from Redshift to S3 and run an ETL job to convert the data type to CSV from Parquet.
C. Use AWS Database Migration Service with the Schema Conversion Tool to convert the data schema.
D. Use AWS Lake Formation.

7. You have a large amount of data stored in a PostgreSQL database on EC2 that you want to migrate to Amazon Aurora. What AWS service would you use for this?

A. AWS Lambda
B. AWS Database Migration Service
C. AWS Data Pipeline
D. AWS Glue

8. You have data incoming from a number of IoT devices that are deployed on your factory floor. You want to take the inputs from these devices and run some interactive SQL queries on the data to power a real-time dashboard as well as set up alarms to alert admins when there is an issue. What AWS streaming services can you use to build this solution?

A. Use Kinesis Firehose to ingest the data and output to ElasticSearch. Run a real-time dashboard using Kibana on top.
B. Use Kinesis Data Streams to ingest the data and use EC2 as a consumer to run SQL queries. Send the outputs to these queries to a custom dashboard application.
C. Ingest streaming data with Kinesis Data Streams but use Kinesis Data Analytics to run SQL queries over windowed streaming data. Create alerts on the incoming data stream as well. Send the outputs to Kinesis Data Streams to output to a DynamoDB table. Build your dashboard on the DynamoDB table.
D. Ingest the data into S3 using S3 copy APIs. Use Amazon Athena to run SQL queries on the S3 data and send the outputs to a custom dashboard built using QuickSight.

9. You have performed OCR on textual data and extracted the outputs in JSON format. The outputs consist of key/value pairs as well as some raw text fields. What kind of data is this?

A. Structured data
B. Semi-structured data
C. Unstructured text data
D. Unstructured image data


10. You have split video frames into images and stored the images in S3 for further processing.
Your ML scientists want to build image classification models on top of this image data
directly in JPEG format. What kind of data is this?

A. Structured image data
B. Semi-structured image data
C. Unstructured text data
D. Unstructured image data

11. You are given a dataset that consists of a day_of_week field. What kind of feature is this?

A. Label
B. Continuous feature
C. Categorical feature
D. Data point

12. You have a large amount of streaming video data incoming from various video feeds you
have in place. You need to quickly ingest this video data and perform some image analysis
and object detection on the video. What architecture might you propose that the customer
use in this use case that will minimize the amount of code the customer needs to manage?

A. Ingest the video using Kinesis Data Streams. Build an ML model to do object and scene
detection using Amazon SageMaker, and host the model using EC2. Set up an EC2
consumer to process the video.
B. Ingest the video using Kinesis Data Streams. Build an ML model to do object and scene
detection using Amazon SageMaker, and host the model using Lambda. Set up a Lambda
consumer to process the video.
C. Ingest the video using Kinesis Video Streams. Build an ML model to do object and scene
detection using Amazon SageMaker, and host the model using SageMaker Endpoint. Set
up a SageMaker consumer to process the video.
D. Ingest the video using Kinesis Video Streams. Use Amazon Rekognition Video as a
consumer to use prebuilt ML models to process the video.

=========================================================================
6
=========================================================================

1. You are building an ML solution to extract entities from financial documents submitted
to the Securities and Exchange Commission (SEC). These entities will be fed into an entity
recognition model. You have collected the relevant SEC filings and stored them on S3. To
prepare the data for your ML model, you need to first label the entities. You have experts in
your company but do not want to build a labeling tool. You have asked your AWS Solutions
Architect to give you some guidance. What guidance should the Solutions Architect give you?

A. Use Redshift ML.
B. Use SageMaker Ground Truth.
C. Use Glue Data Labeler.
D. AWS does not offer any data labeling tools. You have to build your own on premises.

2. You have a large amount of data on S3 and want to run queries on exabyte-scale data. You
are using Redshift as your data warehousing solution already, but you are concerned with the
added cost of moving this data to Redshift. What tool should you use to run SQL queries on
the S3 data without moving the data?

A. Nothing; you always have to move the data out of S3 to run queries.
B. Use Amazon Athena so you don’t have to move the data out of S3.
C. Use Redshift Spectrum. The data does not leave S3, so no additional storage costs are
incurred.
D. Move the data to HDFS on an EMR cluster. Then use Presto on EMR.

3. You need to train a deep learning image model to recognize images of products in your warehouse and identify any malformed or faulty packages. However, you don’t have any labeled
data. You can start labeling data using a public workforce but are concerned about the cost
for labeling hundreds of thousands of images for training an ML model. What advice should
your AWS Solutions Architect give you to circumvent this issue?

A. Use SageMaker Ground Truth with Amazon Mechanical Turk. Wait a few months for all
images to be hand labeled.
B. Train an ML model on SageMaker using a handful of labels. Turn this model on to label
the rest of the images. Retrain the model periodically to ensure that it is seeing more
data.
C. Use SageMaker Ground Truth with Amazon Mechanical Turk. Turn on automated
labeling to speed up labeling of high-confidence images, sending only low-confidence
images to human reviewers.
D. Train an ML model using Spark ML on EMR using a handful of labels. Turn this model
on to label the rest of the images. Retrain the model periodically to ensure that it is seeing more data.

4. You need to run some large-scale distributed ETL jobs on EMR and run some machine learning models on top of that processed data. Which EMR tool is ideally suited for running in memory-distributed computing and for ML workloads on that data?

A. SageMaker Processing and SageMaker
B. Pig for distributed processing and Spark ML for ML
C. HBase for distributed processing and HiveML for ML
D. Spark for distributed processing and Spark ML for ML

5. Which of the following EMR tools is an interactive web application that allows users to author code, equations, text, and visualizations?

A. Jupyter Notebook
B. Hue
C. Ganglia
D. Presto

6. You have trained an ML model for fraud detection and have hosted that model on SageMaker. However, your customer data is stored in Redshift. The users who need to perform downstream analytics on the fraudulent cases are not proficient in ML, but they are familiar with Standard SQL, and you are concerned about the cost of engineering a custom UI and inference service for these users. What solution might you employ to make the predictions of the ML model usable by the end customers?

A. Have your customers call a Lambda function that runs the inference on the SageMaker
model. Import the inferences into Redshift.
B. Replace the SageMaker endpoint with SageMaker batch transform. Store the outputs of
the model in S3 and use the COPY command to load the data into Redshift from S3.
C. Replace the SageMaker endpoint with SageMaker batch transform. Store the outputs of
the model in S3 and use the UNLOAD command to load the data into Redshift from S3.
D. Use Redshift ML. Analysts can write SQL queries that call the SageMaker model using
SQL and generate inferences on the fly directly within Redshift itself.

7. Your data engineers have authored some extract-transform-load (ETL) code using PySpark. Given that you are a small startup with a lean engineering team, you don’t want to worry about managing a persistent Hadoop environment, so you are interested in implementing a serverless ETL solution. However, you also want the solution to be event driven and triggered whenever new data lands in your raw data bucket. What AWS tools might you use to build such an architecture? (Choose all that apply.)

A. Write a Lambda function that calls AWS Glue APIs to kick off a Glue job using the code
authored by your engineers. Set the Lambda to be triggered by an S3 prefix.
B. Write a Lambda function that sets up an EC2 instance from a snapshot containing the
code and associated dependencies authored by your engineers. Set the Lambda to be
triggered by an S3 prefix.
C. Set up an ephemeral EMR cluster to run the job. The cluster will automatically spin
down once the processing is completed.
D. Use a SageMaker Processing job to run your ETL scripts in a serverless manner. Trigger
the SM Processing job using AWS Lambda.


8. Consider the same use case as in the previous question. You have chosen to go with AWS
Glue for ETL, and the processed data is staged in S3. Data scientists would like to have a
UI that can import the data from S3 in tabular format, visually inspect the data, run simple
transformations, and feature engineering steps to produce prepared data for ML. What tool
would you recommend?

A. Build a custom data viewer on Amazon EC2 and use Amazon Athena to write SQL
queries to transform the data.
B. Use Glue Data Brew.
C. Import the data into SageMaker Data Wrangler.
D. Build a custom UI on Amazon EKS to inspect the data and use SageMaker Processing to
run your data transformation logic.

9. Which Apache tool would you use to perform event-driven computations on incoming data streams from clickstream data, log data, or IOT data?

A. Apache Spark
B. Apache Flink
C. Hue
D. Hive

10. You have built fraud models on premises using data stored in the Hadoop Distributed File System (HDFS) that you are now migrating to S3. However, you are under a deadline to deliver the models in production in the cloud and would like to minimize as much code refactoring as possible. The models are trained using PySpark using the pyspark.ml and mllib libraries. What suggestion would you provide to the customer as their AWS Solutions Architect?

A. Move their code and dependencies into a custom docker container and use Amazon
SageMaker for training ML models since SageMaker is the leading ML platform on
AWS.
B. Given the tight timelines, set up an EC2 cluster with the PySpark code and dependencies
to run your training job.
C. Transform their code to PyTorch instead and run the model training on SageMaker since
PyTorch is a growing open-source framework for ML.
D. For the time being, given that the customer is under a deadline, spin up an EMR cluster
leveraging existing code to train the models.

=========================================================================
7
=========================================================================


1. You are a company that is just starting out in machine learning and need to engineer features on a large dataset. However, your team currently is composed of analysts who lack data engineering skill sets such as PySpark and Scala. What tools would you recommend they start with to prepare data and engineer features? (Choose all that apply.)

A. SageMaker notebooks
B. Amazon EMR
C. AWS Glue Data Brew
D. SageMaker Data Wrangler

2. You are a company that is just starting out in machine learning and need to engineer features on a large dataset stored in an Amazon Aurora database. Which AWS tool could you use to visualize the data and engineer features with little to no code?

A. Jupyter Notebooks on EMR
B. AWS Glue Data Brew
C. Spark Cluster on EMR
D. SageMaker Data Wrangler

3. You are a data scientist and your engineering team has provided you with a dataset to build
ML models on, but one of the columns has 30 percent of the values missing. What strategy
would you use to fill in the missing dataset to avoid introducing bias in the data?

A. Nothing; just remove the column.
B. Replace with the most frequent value to reduce the bias.
C. Replace with the mean.
D. Train an ML model to predict the missing values.

4. Which of the following is a valid data engineering strategy?

A. Split raw dataset into train/test ➢ standardize training dataset ➢ use the mean/standard
deviation values to standardize the test set ➢ train the model on the training data and
make predictions on the test set.
B. Standardize entire dataset ➢ split standardized dataset into train/test ➢ train the model
on the training data and make predictions on the test set.
C. Split raw dataset into train/test ➢ convert categorical values in the training dataset alone
to one-hot encoded values ➢ train the model on the training data and make predictions
on the test set.
D. Split raw dataset into train/test ➢ use a mean imputer to impute missing values with the
mean in the training dataset ➢ train the model on the training data and make predictions
on the test set with some missing values.


5. You are a data scientist at a bank working on a fraud detection problem. You notice that the number of fraudulent examples in your dataset is extremely small. You are concerned that your model will not be able to detect fraudulent examples well due to the lack of examples in the training data. What are some techniques you can apply to mitigate this? (Choose all that apply.)

A. Write a custom loss function that penalizes the model more for incorrectly predicting
fraudulent examples.
B. Perform SMOTE upsampling on the fraudulent class.
C. Upsample the fraudulent class.
D. Simply drop the minority class since you have too few examples and the model will not
be able to learn from them.

6. Which of the following are valid data augmentation strategies for image labeling use cases? (Choose all that apply.)

A. Use SageMaker Ground Truth to label more images for use in your ML model.
B. Duplicate each image by converting the format from PNG to JPEG.
C. Rotate each image by a random angle.
D. Stem the images.

7. You would like to build an image classification algorithm but you are concerned with the lack of labeled data and the expense and time in labeling data. In order to convince leadership of the need for new data, you need to create a baseline model with good performance. What simple strategy can you use to generate more labeled data?

A. Use a log transform data augmentation strategy.
B. Use SMOTE to upsample the images.
C. Augment the images by rotating, cropping, and changing the RGB contrasts to generate
more labeled images to train a more robust model.
D. Use an external third-party vendor to generate more labeled data. There is no other way.

8. A data scientist has noticed that one of the columns in her tabular dataset has over 20 percent missing values. Upon plotting the data distribution, she finds that the distribution is skewed and non-normal, with several outliers. What strategy should she consider to treat the
missing values to quickly start building an ML model?

A. Use a mean imputation strategy.
B. Use SMOTE to impute missing values.
C. Drop the rows containing missing values; it won’t affect the model performance much.
D. Use median imputation.


9. A data scientist has engineered features, normalized the data using a standard scaler, and then split the data into train, test, and validation. The modeler is getting near 100 percent accuracy on training, test, and validation datasets and is very satisfied. As the lead data scientist, you are asked to approve the model, but what may the data scientist have missed to get such good performance?

A. The data scientist should have used a MinMaxScaler normalizer.
B. A standard scaler is only useful if the feature is normally distributed.
C. Nothing; you should approve this model for production.
D. The scientist should have split the data into train, test, and validation before normalizing
the data, not after.

10. A data scientist is attempting to improve his model performance. He has observed that the
performance appears to be insensitive to normalizing the features; however, he believes that
normalization is important as the features have very different scales. What might be true of
the model?

A. The model is likely unsupervised; unsupervised ML is insensitive to feature normalization.
B. The model is likely a linear model; linear models are insensitive to feature normalization.
C. The model is likely a tree-based model, which are typically insensitive to feature normalization.
D. All ML models are insensitive to normalization, and as such, it is not a particularly useful feature engineering technique.

=========================================================================
8
=========================================================================

1. A customer is getting started with using deep learning to classify sports video clips. Video clips are tagged using a voting system open to the public on a website. The customer is confused as to what kind of problem this is. What do you tell the customer?

A. This is an unsupervised learning problem since the tagging is not supervised by the customer.
B. This is a supervised learning problem since tags from the community can be used as
labels.
C. This is a reinforcement learning problem since the website can be used as an environment that human agents interact with.
D. This is a semi-supervised learning problem where the videos are first clustered into similar videos, followed by using a subset of the tags for classification.

2. You are generating an ML model for your team using Amazon SageMaker. Your team has a fixed cost budget on the total number of training jobs you can experiment with, but you still want to explore multiple parameters and deliver the best model to the business. What hyperparameter optimization (HPO) strategy will you use for this?

A. Since the customer is on a budget, just experiment with a few sets of parameters manually
B. Random search
C. Bayesian optimization
D. Search engine optimization (SEO)

3. You have a sparse dataset with hundreds of feature columns (X) and a predictor column (y) with numbers. The dataset also contains a significant portion of outliers. You have a simple linear regression model that is not performing well, and you think many of the features are not important. One of your colleagues asks you to try regularization. What type of regularization will you try first?

A. L3
B. ElasticNet
C. L2
D. L1

4. A customer has a dataset with millions of rows and is interested in using XGBoost to predict the value of a target variable. The dataset contains several missing values. What advice will you give the customer before she starts training? (Choose all that apply.)

A. XGBoost cannot handle missing values.
B. XGBoost handles missing values by treating each missing value as NaN.
C. XGBoost supports missing values by default, and branch directions for missing values
are learned during the training process.
D. XGBoost supports missing values by treating missing values as 0 in some cases.


5. You have a dataset containing images of printers placed on tables. What built-in SageMaker algorithm can be used to identify the location of the printer with respect to the table?

A. Object detection
B. Image classification
C. DeepAR
D. Object2Vec

6. A customer would like to use historical sales data for predicting future sales of several items
on their inventory for planning purposes. What services on AWS can help with this problem?
(Choose all that apply.)

A. Amazon Forecast
B. DeepAR built-in algorithm on Amazon SageMaker
C. Amazon Personalize
D. AWS Lambda

7. You have a large language model that needs to be fine-tuned based on a custom dataset. The
instances that you have access to cannot fit the model in memory, but you have access to
multiple instances. Which of the following distributed training strategies will you use?

A. Data parallel
B. Model parallel
C. Device parallel
D. Instance parallel

8. Your team currently runs ML training on notebooks with the assumption that running these
jobs locally on a notebook gives them access to logs. These logs are useful for fine-tuning and
debugging their models. They feel that using a service like SageMaker will prevent them from
having access to logs. What advice will you give them?

A. Continue using the notebook; their assumption is correct.
B. Training job logs can be obtained by contacting AWS.
C. SageMaker training job logs can be found in CloudWatch logs.
D. SageMaker training job logs can be found in CloudTrail logs.

9. Which of the following is the easiest way to react to a training job status change on Amazon
SageMaker and trigger an action based on this change?

A. Use EventBridge.
B. Poll for the job status using a Lambda.
C. Poll for the job status using an EC2 instance.
D. Use SageMaker Debugger.

10. A customer archives news articles produced by major publications and would like to use
machine learning to summarize these articles into short sentences. Manually summarizing
each article produced so far will be impractical and costly. What built-in algorithm with
Amazon SageMaker will you suggest the customer try out for this use case?

A. SageMaker Summarizer
B. SageMaker Sequence-to-Sequence
C. SageMaker NTM
D. SageMaker BlazingText

11. Consider the same use case as in the previous question. You have used Ground Truth to label
the helmets on a subset of players. What algorithm would you use next to train a model to
detect helmets on players?

A. Object detection using Single Shot Detector (SSD)
B. Image classification using Inception
C. Random Forest
D. ARIMA

=========================================================================
9
=========================================================================


1. You are tasked with compiling a list of all training jobs with high-level metrics in a table. What typical APIs would you use to do this for AI/ML services on AWS?

A. list- and describe- APIs
B. get- and describe- APIs
C. getlist- and describelist- APIs
D. create- and describe- APIs

2. A customer would like to track and compare different SageMaker trials involving training and batch transform jobs. What SageMaker feature(s) can help with this? (Choose all that apply.)

A. SageMaker Neo
B. SageMaker MLtracker
C. SageMaker Experiment
D. SageMaker Search

3. A model you trained has the following training (solid line) and validation (dashed line) curves (see the following graphic). What term describes your model correctly?

Loss
epochs
A. Underfitting
B. Overfitting
C. Diverging
D. None of the above

4. A customer has a time series dataset that is used to train a forecasting model using Amazon
Forecast. Which of the following metrics should they choose to evaluate and compare multiple forecasting models? (Choose all that apply.)

A. Root mean squared error (RMSE)
B. Weighted quantile loss (WQL)
C. F1 score
D. Recall

5. A two-class classification model that you have trained results in the following confusion
matrix. You are tasked with calculating the false positive rate since subject matter experts
in your company need it to be as low as possible. What is the value of the false positive rate
based on this confusion matrix?

Class 1 Class 2
Class 1 TP = 50 FP = 5
Class 2 FN = 2 TN = 5
A. 10
B. 0.1
C. 0.5
D. 0.25

6. An image classification algorithm is used to distinguish between benign and malignant
tumors detected on X-ray images of patients. Your company wants to correctly detect
all patients (100 percent) who actually have cancer. What metric should you use to tune
your models?

A. Accuracy
B. Squared error
C. Recall
D. Precision


7. A customer has a dataset with two features, X1 and X2. When plotted on a graph, this customer’s dataset looks like the following graphic. The customer would like to distinguish two types of rows (labeled as “o” and “x” in the graph) in the dataset by building a neural network. What advice will you give the customer to help with this use case?

A. Neural networks can be used for image classification jobs but not for classification of
numerical data in a tabular dataset.
B. Neural networks can only be used for regression and forecasting use cases but not for
classification.
C. Build a random forest classifier using neural networks to classify this dataset.
D. Neural networks can be used to train this model. Tell the customer to go ahead and try
building and training a model.

8. You want to create a model that can classify fraudulent credit card transactions. What
objective metric will you use to determine how good your model is on Amazon SageMaker
Autopilot?

A. AUC (area under the ROC curve)
B. F1 score
C. Number of false negatives
D. Number of false positives


9. A company’s employees have recently been receiving many spam and phishing emails. The
company has a machine learning team that would like to explore AWS services that can help
analyze the subject line and decide whether an incoming email is spam. Spam emails will be
immediately deleted and not forwarded to employees. After some exploration, the ML team
has decided to try Comprehend Custom for this purpose. What metric should the ML team
use to evaluate their model?

A. Number of spam emails not detected
B. F1 score
C. Number of spam emails detected
D. Number of normal emails detected

10. An ML experiment management system lets you compare multiple training jobs by creating interesting visual plots. You have been using this system to build hundreds of gradient boosted trees using a popular algorithm for the past month and want to look at the impact of changing two hyperparameters—maximum depth and learning rate—on the validation accuracy of the model. To this end, which of the following methods will provide you with the most amount of information that can help you analyze your experiments quickly with the least amount of work?

A. Create a scatter plot with the max depth and learning rate as the axes, and color each
point by the validation accuracy.
B. Create a bubble plot with the max depth and learning rate as the axes and have the size
of the bubble represent the validation accuracy.
C. Create a line plot with two y-axes, one for max depth and another for learning rate;
have the validation accuracy on the x-axis.
D. Create a bar chart for each trial representing the validation accuracy.

=========================================================================
10
=========================================================================

1. A customer that is currently exploring Amazon Kendra recently read that Amazon SageMaker provides users with several options to host, maintain, and update endpoints. The customer is wondering if Kendra models can be hosted on SageMaker. What advice will you give them?

A. It is easy to host Amazon Kendra models on SageMaker with one click on the console or
with any supported API.
B. You cannot host Kendra models on SageMaker.
C. You will first need to export your Kendra models, and import the same in SageMaker
for hosting.
D. SageMaker APIs can connect to various AI-level services, including Kendra.

2. A customer would like to deploy a model trained on one of their on-premises workstations on AWS. What is the easiest way to do this?

A. Build a model-hosting service using EC2 instances and upload their model to this custom
service.
B. Use SageMaker training to retrain their model on AWS, and then host the same using
EC2 instances as an endpoint.
C. Use SageMaker training to retrain their model on AWS and then host the same using
SageMaker endpoints.
D. Use the trained model as is to deploy an endpoint on Amazon SageMaker.

3. A customer has trained a recommendation system using Amazon Personalize and needs a way to provide engineers on her team with an API endpoint for generating recommendations for users visiting their e-commerce website. What is the easiest way to provide this functionality to her engineering team?

A. Use a SageMaker endpoint.
B. Create a Personalize campaign from the trained solution.
C. Create a Personalize solution from the trained campaign.
D. Use API Gateway and Lambda.

4. A customer with a production SageMaker endpoint observes that traffic to the endpoint has
been increasing rapidly over the past few weeks. What should the customer do to successfully
serve users in the weeks to follow? (Choose all that apply.)

A. Use Production variants.
B. Use a new endpoint with a larger instance type.
C. Set up autoscaling.
D. Block traffic to the new customers.

5. You would like to release a new version of a web application that points to a more accurate machine learning model behind an endpoint. You have selected a small group of 20 beta testers and have also set up a group of simulated users to test out this new functionality. What testing strategy is appropriate to use in this case?

A. A/B testing
B. Canary testing
C. Blue/green testing
D. Red/black testing

6. You are rolling out a new service for external clients. In order to quickly test and roll back changes, you want to deploy the service to a small subset of servers first before rolling it out broadly to production. You also want this system to provide early alarms before customers are impacted. Which of the following test strategies would you use?

A. Canary release
B. Rolling release
C. Blue/green release
D. Shadow release

7. You are rolling out major updates to an existing ML-based service for external clients. You have a model in production that is currently serving traffic and a separate model in a different AWS account and environment that you are currently using to run user acceptance testing and QA. Once QA testing is complete, you will switch all traffic to the new model. Which of the following release strategies does this correspond to?

A. Canary release
B. Rolling release
C. Blue/green release
D. Shadow release

8. Which of the following statements is true regarding SageMaker model hosting?

A. SageMaker model hosting supports autoscaling to elastically scale in or out depending on the incoming client traffic.
B. SageMaker model hosting allows clients to make requests over HTTPS.
C. SageMaker model hosting requires a model.tar.gz artifact in S3 and an inference container in ECR.
D. All of the above options are correct.

9. Which of the following statements is true regarding SageMaker Batch Transform? (Choose all that apply.)

A. SageMaker Batch Transform supports autoscaling to elastically scale in or out depending
on the incoming client traffic.
B. SageMaker Batch Transform creates a persistent instance or cluster that remains online
even after all the inferences have been computed.
C. SageMaker Batch Transform requires a model.tar.gz artifact in S3 and an inference container in ECR.
D. SageMaker Batch Transform uses an agent to distribute the incoming dataset across multiple hosts and collate the results before sending them back to Amazon S3 for storage.

10. For which of the following services is the model hosting managed by AWS and not controlled by the customer? (Choose all that apply.)

A. Amazon Rekognition
B. Amazon Kendra
C. Amazon Fargate
D. Amazon SageMaker

=========================================================================
***** 11
=========================================================================

1. Your client has hired you to take one of their models trained on SageMaker and deploy
it on iOS devices. The application is expected to run without any Internet connection on
tens of thousands of iPhones. Which of the following steps will you take to achieve this for
your client?

A. You cannot host SageMaker models on iOS devices.
B. Host the SageMaker model on the cloud and use API calls to get model predictions.
C. Use SageMaker Neo to create an Apple CoreML model and use this in the iOS
application locally.
D. Use SageMaker Edge Manager to host these models with one click onto tens of
thousands of iOS devices.

2. Your team of front-end developers does not have experience with building, training, and
deploying ML models but would like to use natural language understanding features. The
application needs to run on the web as well as on mobile devices. Which of these options will
you ask your team to explore as the easiest first step?

A. Use SageMaker front-end libraries to create custom NLU solutions.
B. Use Amplify and built-in NLU functionality with front-end libraries of your choice.
C. Convert your models to Tensorflow.js-compatible ones, and build a Progressive web
app to deploy these models.
D. Train your team on topics around NLU.

3. Your company has sensitive data on-prem that business teams would like to use for machine learning. However, the teams would like to use AWS AI/ML services to deploy these models. Which of these options makes sense to implement for multiple business teams?

A. Keep the data on-prem; train models using open source frameworks with on-prem
compute, and use Amazon SageMaker to host your models.
B. Amazon SageMaker can connect to all kinds of on-prem data storage solutions. You can
keep your data on-prem and train and deploy models on the cloud.
C. Use Direct Connect to transfer data to Amazon S3; then train and deploy your models
using SageMaker.
D. Use the AWS transfer family to transfer data to Amazon S3; then train and deploy your
models using SageMaker.

4. Due to recent changes in security policies, EKS-based ML applications that you were building
and using on the cloud have to be migrated to on-prem compute. Which of the following
options will you use to migrate these ML applications to on-prem?

A. Keep the data on-prem; train models using open source frameworks with on-prem
compute, and use Amazon SageMaker to host your models.
B. Use EKS Anywhere and the same EKS tooling on-prem for your ML applications.
C. Use containers and container definitions with custom tooling on-prem.
D. There is no way to migrate these applications to on-prem.


5. You would like to expose your SageMaker model as a hosted endpoint to certain end users with registered API keys. Which of the following patterns presents the simplest way to achieve this?

A. SageMaker endpoints are public by default. You can simply post inference payloads and
receive a prediction response.
B. This is not possible using SageMaker; host your models using AWS Lambda and integrate with API gateway.
C. Directly integrate API Gateway to SageMaker using custom mapping templates.
D. Create a CloudFront distribution. Then use a container-based Lambda function and host
your SageMaker model using Lambda@Edge.

6. Due to the sensitive nature of the data used by your company, you need to host ML models
that are trained on such data on premises. The models are trained using Amazon SageMaker
on the cloud. What approach would you use to deploy the model on premises?

A. Use Amazon SageMaker Anywhere to deploy the model locally.
B. Export the model artifact to your on-premises cluster and host it on premises.
C. You cannot export the SageMaker Model artifact on premises, so you will have to host
the model in the cloud.
D. Use SageMaker Edge Manager to host the model on premises.

7. In order to build a decoupled architecture, you are hosting your ML model trained on SageMaker behind an API gateway. However, the incoming client POST requests need to be transformed to CSV format before the model can make predictions. What AWS service might you
use to preprocess the incoming data before calling the endpoint serving API?

A. Amazon EC2
B. AWS Lambda
C. Amazon SageMaker Processing
D. Amazon EKS

8. You are an ML engineer at a growing SaaS provider and need to host thousands of deep learning models into production on AWS. You have read through a number of different potential options, but a key consideration when hosting these models is the cost of the underlying hosting infrastructure and the low round-trip (under 100 ms) latency for serving. Some of the models need to be hosted on CPUs, whereas others may require GPUs. What hosting architecture might you use to save on cost?

A. Create thousands of SageMaker endpoints to host your models.
B. Use a SageMaker multi-model endpoint for hosting your models. You can host multiple models on the same machine, thus lowering cost.
C. Build a custom hosting service using Amazon EKS. You can use bin packing to efficiently pack multiple containers on the same host, achieving both low cost and lower latency.
D. Use AWS Lambda to host your models.


9. You are an ML engineer at a growing SaaS provider and need to host a small model that will
serve inferences with high throughput in production. Serving latency is a concern, and you
would like to avoid cold start. However, you are also concerned with the cost of spinning up
a large instance to meet the high-throughput needs for a small model. As the AWS solutions
architect, you might propose which of the following architectures? (Choose all that apply.)

A. Host the small model on a t3.medium SageMaker model hosting endpoint to save on
cost.
B. Host the model on Amazon S3 to save on cost.
C. Host the model on several small t3.medium SageMaker model hosting endpoints to meet
the throughput needs and to keep costs low.
D. Use AWS Lambda to host the model since it is small, and use provisioned concurrency to
avoid cold start.

10. You are the AWS solutions architect for a growing marketing firm that is looking to quickly ingest customer data, segment consumers, and send out personalized content to them via SMS and push notifications. The company is using Amazon Personalize to generate the recommendations. The marketing team is looking for your advice on how to build such a system in the easiest possible way. What solutions would you recommend?

A. Do nothing. Amazon Personalize offers a service to directly send email and push notifications to customers with recommended content.
B. Use Amazon SES and SNS to send out emails and push notifications and use AWS
Lambda to write customized content for each user segment.
C. Use Amazon SQS and SNS to send out emails and push notifications and use AWS
Lambda to write customized content for each user segment.
D. Use Amazon Pinpoint to send customized and personalized content to users across multiple communication channels.

=========================================================================
**** 12
=========================================================================


1. Your ML platform team would like to reproducibly deploy ML environments for different
lines of business (LoBs) to use in a reliable manner. What AWS tool can be used to build the
underlying infrastructure as code (IaC)?

A. Amazon SageMaker Pipelines
B. Amazon CloudWatch
C. AWS CloudFormation
D. AWS Step Functions

2. Your ML platform team would like to build a manual human approval step into the ML release pipeline to validate that the model performance meets business requirements. What tool can be used to achieve this in the easiest way?

A. AWS CodeCommit
B. AWS CodeBuild
C. AWS Step Functions
D. AWS CodePipeline

3. Which AWS service would you use to build model containers from source code as part of your CI process?

A. AWS CodeBuild
B. AWS CodeCommit
C. AWS CodeDeploy
D. Amazon Elastic Container Registry

4. You have successfully deployed an ML model on SageMaker to production that deals with customer clickstream data. Your management is concerned that the model may start to perform poorly over time as customer preferences change. What recommendation would you give them?

A. Do nothing; ML model accuracy does not degrade over time.
B. Use Amazon CloudWatch to monitor the underlying compute infrastructure to ensure
that the model can service incoming client traffic.
C. Set up model monitoring on the endpoint to detect data drift. Once the drift passes a
threshold, model monitoring will report this in CloudWatch. Set up alarms to be alerted
when drift is detected.
D. Retrain and redeploy your models every week.


5. What are the right steps in the SageMaker model monitoring workflow?

A. Set up an endpoint with data capture. SageMaker will automatically monitor the endpoint at an hourly frequency and detect drift.
B. Deploy a SageMaker endpoint with Data Capture ➢ Run a baseline job for model monitoring to generate a baseline of the input feature statistics ➢ Set up model monitoring by passing in the baseline statistics and the endpoint as input and deciding on a monitoring interval (daily, weekly, hourly, etc.).
C. Deploy a SageMaker endpoint without Data Capture ➢ Run a baseline job for model monitoring to generate a baseline of the input feature statistics ➢ Set up model monitoring by passing in the baseline statistics and the endpoint as input and deciding on a monitoring interval (daily, weekly, hourly, etc.).
D. Source data ➢ Explore data ➢ Engineer features ➢ Train a model.

6. You are using AWS CloudFormation templates to create and manage the VPC and IAM roles and policies, as well as the SageMaker Studio domain for data scientists to use for development using notebooks. You have created and tested the template and now would like to deploy templates across several target accounts spread across multiple AWS regions. What tool would you use to automate this?

A. AWS CloudFormation templates can be automatically deployed across regions and
accounts. No extra work is needed.
B. Use AWS CloudFormation StackSets to automate cross-region and cross-account deployments.
C. AWS does not offer a tool for this. You need to manually deploy the CloudFormation
template across different accounts in different regions.
D. Use AWS CodePipeline to deploy the CloudFormation templates across different
accounts.

7. Your company has a new CTO, who has instituted a mandate to leverage infrastructure-ascode tools like Terraform wherever possible to manage infrastructure deployments. Your ML infrastructure team is proficient in programming languages like Python but less familiar with expressing intent directly using JSON/YAML. What AWS tool might you use to simplify the creation of infrastructure-as-code templates using familiar programming languages?

A. Use Jupyter Notebooks to author the templates.
B. Use AWS Lambda to author the templates.
C. Use AWS Cloud Development Kit (CDK) for Terraform (cdktf).
D. You cannot author Terraform templates using CDK. You need to use JSON/YAML.


8. Your ML engineers are training hundreds of small models across different platforms (SageMaker, EC2, ECS, EMR, etc.) that need to be reliably deployed to production. Errors in deployments must be caught, the deployment needs to be retried or rolled back, and users have to be notified. Administrators and engineers also require a visual tool to view the execution of the pipeline to quickly identify and visualize failed steps. What deployment tool might you recommend that the customer use?

A. Use AWS Step Functions.
B. Use AWS SageMaker Pipelines.
C. Use KubeFlow pipelines.
D. Use AWS CodePipeline.

9. You are using SageMaker Pipelines to train, tune, and deploy a SageMaker model. Which of
the following steps would you use to test whether the model performance exceeds a previous
threshold and branch out the action based on the output?

A. Transform Step
B. Training Step
C. Callback Step
D. Condition Step

10. Which AWS service would you use to check in and check out code, set up version control,
maintain commits, and trigger pipelines in response to code commits made to a mainline branch?

A. AWS CodeBuild
B. AWS CodeCommit
C. AWS CodeDeploy
D. Amazon Elastic Container Registry

=========================================================================
***** 13
=========================================================================

1. Which of the following is AWS’s responsibility when it comes to security of Amazon SageMaker? (Choose all that apply.)
A. Patching of the instances used to run SageMaker Training jobs
B. Applying OS-level updates to the instances used to run SageMaker notebooks
C. Managing the S3 bucket where SageMaker publishes the model artifacts after training
the model
D. Managing the network of your VPC, which grants SageMaker access to your AWS
resources

2. Which of the following is the customer’s responsibility when using Amazon Comprehend?
A. Patching the EC2 instances that host the Comprehend custom model
B. Managing the EBS volume attached to the training instances used to train the Comprehend custom classifier
C. Managing the health of the endpoint used by Comprehend DetectSentiment API
D. Providing customer-managed CMKs to Comprehend to encrypt any data stored by the
service at rest

3. Which AWS service would you use to view performance metrics from your SageMaker endpoint and set alarms and appropriate thresholds to autoscale the SageMaker endpoint if they are surpassed?

A. Amazon Inspector
B. Amazon CloudWatch
C. AWS Config
D. AWS CloudTrail

4. Which AWS service would you use to log API calls made by users to Amazon SageMaker in
their machine learning environments for audit purposes?

A. AWS GuardDuty
B. Amazon CloudWatch
C. AWS Config
D. AWS CloudTrail

5. You have a requirement to use customer-managed CMKs for all training jobs launched by the
user to encrypt the EBS volume attached to the SageMaker training instance. What setting is
missing from the following API call that the data scientist needs to include?

sagemaker.estimator.Estimator(
 image,
 role,
 instance_count=1,
Review Questions 239
 instance_type='ml.m4.xlarge',
 max_run=3600,
 output_path='s3://{}/{}/models'.format(output_bucket, prefix),
 sagemaker_session=sess,
 max_wait=3600,
 encrypt_inter_container_traffic=False
)

A. Nothing; SageMaker automatically uses a customer-managed CMK to encrypt attached
storage.
B. Add an output_kms_key parameter and pass in your customer-managed CMK.
C. Add a volume_kms_key parameter and pass in your customer-managed CMK.
D. SageMaker does not support customer-managed CMKs for encryption at rest.

6. You have a requirement to only launch SageMaker training jobs inside your VPC because
your S3 buckets are not accessible over the public Internet. What setting is missing from the
following API call that the data scientist needs to include?

sagemaker.estimator.Estimator(
 image,
 role,
 instance_count=1,
 instance_type='ml.m4.xlarge',
 max_run=3600,
 output_path='s3://{}/{}/models'.format(output_bucket, prefix),
 sagemaker_session=sess,
 max_wait=3600,
 encrypt_inter_container_traffic=False
)

A. Provide your network settings using the security_group_ids and subnets parameters.
B. Provide your network settings using the security_group_ids and subnets parameters. A minimum of 2 subnets in different AZs are required.
C. Expose your S3 bucket to the public Internet since there is no other way for SageMaker
to access it.
D. SageMaker can automatically connect to your S3 buckets via the AWS network without
any additional configurations on your part.

=========================================================================
14
=========================================================================


1. Which of the following AWS services can you use to discover when the InvocationsPerInstance metric of the SageMaker endpoint exceeds the desired threshold?

A. Amazon CloudWatch Logs
B. Amazon CloudWatch Metrics
C. Amazon CloudTrail
D. AWS Config

2. You are part of an ML engineering team tasked with deploying a fault-tolerant ML application. Which of the following best practices might you consider for storing model artifacts to aid in recovery and failure management? (Choose all that apply.)

A. Consider using S3 Object Lock to implement a WORM model.
B. Store model artifacts in S3 and implement a bucket policy that only allows certain IAM
principals to access the bucket.
C. Use S3 versioning and MFA delete to require multifactor authentication (MFA) when
deleting an object version.
D. Store model artifacts in an EBS volume attached to the host instance for deployment.

3. You are part of an ML engineering team tasked with deploying a highly available ML application on Amazon SageMaker. Which of the following best practices might you consider for ensuring high availability while also enforcing low cost?

A. Deploy the model endpoints using multiple instances for high availability in multiple
regions and in an active-active configuration for disaster recovery.
B. Deploy the model artifacts in a single region but using multiple instances. In the case of
failure, SageMaker will automatically attempt to redistribute instances across availability
zones (AZs).
C. Deploy the model artifacts in a single region but using a single instance. In the case of
failure, SageMaker will automatically attempt to restart the instance in a different AZ.
D. SageMaker endpoints are not highly available. Deploy on EC2 instead and use multiple
instances in different AZs for high availability and low cost.

4. Which of the following services can you use to deploy ML infrastructure using infrastructure as-code practices?

A. AWS CloudFront
B. AWS Elastic BeanStalk
C. AWS CodePipeline
D. AWS CloudFormation


5. Which of the following are advantages of separating production workloads in their own AWS account? (Choose all that apply.)

A. AWS recommends deploying dev, test, and production workloads in the same AWS account to save cost.
B. Fault isolation; separating production workloads in their own account can reduce the possibility of a failure in one system affecting other production systems.
C. Accounts are a hard security boundary in AWS. By separating production workloads in their own account, you can implement least privilege and minimize human access to production workloads.
D. It simplifies disaster recovery when an application fails.

6. You are part of an ML engineering team tasked with deploying a highly available ML application on Amazon SageMaker. Which of the following best practices might you consider for ensuring high availability while also enforcing low cost?

A. Deploy the model on two or more instances across multiple availability zones (AZs) for
high availability and use SageMaker spot pricing to lower cost.
B. Deploy the model artifacts in a single region but using an inf1 instance. In the case of
failure, SageMaker will automatically attempt to restart the instance in a different AZ,
and inf1 instances are lowest-cost instances for inference on SageMaker.
C. Deploy the model artifacts in a single region but using multiple instances. In the case of
failure, SageMaker will automatically attempt to redistribute instances across AZs.
D. SageMaker endpoints are not highly available. Deploy on EC2 instead and use multiple
instances in different AZs for high availability and low cost.

7. You are part of an ML engineering team tasked with storing production ML model metadata
and lineage for reproducibility, auditability, and traceability. Your company has wasted dev
cycles trying to trace back code associated with production models serving external client
traffic when the models were audited externally. Which of the following ML artifacts would
you consider storing as part of an ML lineage or metadata management service?

A. Data the model was trained on.
B. Dependencies used to build the model training and hosting containers.
C. Commit ID or hash to the code used to train the algorithm and deploy the model.
D. All of the above options are correct.

8. You are part of an ML engineering team tasked with storing production ML model metadata
and lineage for reproducibility, auditability, and traceability. Your company has wasted dev
cycles trying to trace back code associated with production models serving external client
traffic when the models were audited externally. Which of the following ML artifacts are
likely not useful for storing as part of the lineage service?

A. Data the model was trained on.
B. Jupyter notebooks used for data exploration and model training.
C. Commit ID or hash to the code used to train the algorithm and deploy the model.
D. Hashes to the containers containing stable dependencies and packages used to produce
the model.


9. In deploying a new ML model to production, you want to test the new model on a control
group of customers to measure any uplift in your business key performance indicators (KPIs)
before rolling the model out in production to all customers. What testing strategy might you
use to achieve this?

A. A/B testing
B. Blue/green testing
C. Canary testing
D. Rolling deployments

10. Which of the following AWS services would you use to view logs from SageMaker Processing jobs?

A. SageMaker Studio
B. SageMaker Profiler
C. AWS CloudWatch
D. AWS CloudTrail

=========================================================================
15
=========================================================================

1. You work for a media company where you are building a computer vision neural network model to identify objects from images. You notice the model is taking a very long time to train even a single epoch on a single CPU. What are some approaches you can take to boost model training speed without sacrificing model performance?

A. Switch to a larger CPU with more memory.
B. Reduce the amount of data sent to the model and train the model on a smaller dataset.
C. Reduce the number of layers in your deep neural network to reduce the number of
training parameters.
D. Switch to a P3 GPU instance type.

2. You have trained a deep learning model on a P3.8xlarge GPU and are now ready to deploy
the model to production. However, you have noticed during testing that GPU usage during
inference is quite low and that 3.8xlarge instances are quite expensive. What are some ways
to optimize your hosting workloads to maintain the high throughput but lower the inference
costs? (Choose all that apply.)

A. Consider running your inference on lower-cost GPUs such as the G4 instance family.
B. Compile your model using the AWS Neuron SDK to be able to run on
EC2 Inf1 instances powered by AWS Inferentia.
C. Switch to using CPUs instead. CPUs will maintain the throughput but significantly lower
the cost.
D. Run inference on AWS Lambda for lowest cost.

3. You have deployed an ML model to production using SageMaker but have begun to notice that during certain times of the day, the inference latency is starting to spike. The spike corresponds to an increase in traffic to the SageMaker endpoint. What is the recommended way to scale your endpoints while maintaining optimal performance and maintaining low costs?

A. Manually deploy the model on a second SageMaker instance and use an application load
balancer to route traffic to either endpoint in a round-robin manner.
B. Switch to a larger instance type to handle the extra demand during certain hours.
C. Use SageMaker Autoscaling to handle demand during peak hours. SageMaker will automatically scale down when the demand drops.
D. Switch from SageMaker hosting to hosting your models on Fargate, and employ ECS
autoscaling on Fargate to handle the increased demand during peak hours.


4. You are training a deep learning model using PyTorch on a large dataset that spans several
hundred gigabytes. In order to train on the large dataset, you have switched to using distributed training across several GPUs. However, you are beginning to notice a significant drop inperformance and are experiencing longer training times. How can you improve the training
performance without sacrificing model performance?

A. Training performance is slowing down due to I/O between nodes. It is expected when
doing distributed training.
B. Switch your framework to TensorFlow, which is known to have better performance than
PyTorch.
C. Modify your training code to use PyTorch Distributed Data Parallel (DDP).
D. Modify your training code to use SageMaker’s Distributed Data Parallelism library.

5. You are working for an ad-tech company and have deployed an ML model to production for
real-time bidding. Your application uses ML to decide the best ad to show upon receiving a
bid request. Due to the real-time nature, your application has very stringent inference latency
requirements in the single-digit milliseconds. What is the best solution for model training and
deployment when faced with this requirement?

A. Train your model on premises but deploy your model using AWS Lambda in the cloud,
fronted with an API gateway for low-latency serving.
B. Train and deploy your model on Amazon SageMaker. SageMaker round-trip inference
latency is already in the single-digit milliseconds.
C. Train your model in the cloud but deploy the model at the edge (at the bidding location)
for real-time, low-latency inference.
D. Train and deploy your model on Amazon EC2. EC2 instances does not have the
overhead of managed services like AWS Lambda and SageMaker and can be customized
to meet your latency needs.

6. You have deployed an ML model in production that has a high throughput requirement for serving incoming client traffic using Amazon SageMaker. You have instrumented your endpoint to monitor the CPU utilization and the memory utilization. What other metric might you consider monitoring in order to ensure that you meet the incoming demand?

A. InvocationsPerInstance
B. ModelLatency
C. GPUUtilization
D. GPU Memory

7. You have deployed an ML model in production that has a high throughput requirement for serving incoming client traffic using Amazon SageMaker. The application is highly latency sensitive, and you need to set up proper thresholds for autoscaling in order to meet the service level objectives (SLOs) for your end customers. During testing, you have instrumented your endpoint to monitor the CPU utilization and the memory utilization. What other metric might you consider monitoring in order to ensure that you correctly set up your autoscaling policy in production?

A. InvocationsPerInstance
B. ModelLatency
C. GPUUtilization
D. GPU Memory

8. You are a machine learning engineer training deep learning models on expensive GPU
instances on Amazon SageMaker. Leadership is concerned about the rising compute costs and
would like to quickly instrument a system to reduce cost for long-running training jobs by
shutting them down if the model performance has not improved after a certain time period.
What AWS service might you consider to build such a system?

A. Have your model code write the loss after each epoch to a STDOUT and store the
results in a file. Use AWS Lambda to read the latest results from the file and if the loss
has not decreased past a set threshold, have the Lambda function stop the training job.
B. SageMaker is a managed service and as such imposes cost premiums. To lower cost,
switch to a self-managed option such as EC2.
C. Use SageMaker Debugger. SageMaker Debugger has built-in rules to monitor algorithm
metrics such as loss and can automatically shut the job down if the loss is not decreasing.
D. Use SageMaker Model Monitor. SageMaker Model Monitor has built-in rules to monitor algorithm metrics such as loss and can automatically shut the job down if the loss is not decreasing.

9. You are part of a large team of machine learning scientists, all of whom need to develop, train, and evaluate ML models. The scientists commit their code to GitHub once they are happy with their algorithm. As the engineering lead supporting these scientists, you are interested in building an automated pipeline where scientists can simply pass in some inputs and run at the click of a button that will train and evaluate the model on the provided dataset so that other scientists can quickly and reliably reproduce each other’s results. What services might you consider to build such a pipeline? (Choose all that apply).

A. AWS CodeBuild
B. AWS CodePipeline
C. KubeFlow
D. AirFlow

10. As part of your enterprise cloud migration, you are starting to test training some models on
Amazon SageMaker. However, your enterprise is using Airflow for CI/CD, as currently they are
operating in a multicloud or hybrid environment. They are reluctant to migrate away from this
because Airflow is used in many applications within the company. The customer is concerned
that SageMaker as a managed service will not run using Airflow. As their trusted AWS solutions
architect, you would give which of the following advice to the customer to build MLOps pipelines?

A. The customer is correct. SageMaker does not run on Airflow, but it does run on KubeFlow. Switch to KubeFlow to still use open-source technology, and use SageMaker Operators for Kubernetes instead.
B. The customer is correct. SageMaker does not run on Airflow, but it does run on KubeFlow. Switch to KubeFlow to still use open-source technology, and use SageMaker KubeFlow pipeline components instead.
C. The customer is correct. SageMaker does not run on Airflow, but it does run on SageMaker pipelines. Switch to SageMaker pipelines for your MLOps needs.
D. Use SageMaker operators for Airflow.


=========================================================================
*******16
=========================================================================


1. Your team currently runs a weekly HPO job and uses grid search for finding optimal hyperparameters for a model. The team realizes that this is very costly. What strategy will you suggest to get to the same or better level of accuracy with a lower cost?
A. Tune a model while changing only half the parameters.
B. Tune a model with random search on SageMaker.
C. Tune a model with Bayesian optimization on SageMaker.
D. Tune a model on a notebook instance with grid search.

2. You want to use spot training on Amazon SageMaker but notice that sometimes the TensorFlow training job is interrupted and you lose progress. Although this happens rarely, you want to continue using spot training without losing work. Which of the following steps will you take?

A. Stop using spot instances and use on-demand instances.
B. Detect when the training job stops due to spot instance interruption and start an
on-demand training job.
C. Detect when the training job stops due to spot instance interruption and start a spot
training job.
D. Use checkpointing and continue progress with training in the new instance that SageMaker launches.

3. You want to label 100,000 images with the least cost and time on AWS. Which of the following options will you use?
A. Use an open source labeling tool hosted on AWS and labeling.
B. Use SageMaker Ground Truth with the auto-labeling feature with a Mechanical Turk
workforce.
C. Use Rekognition custom labels to label your images.
D. Use Amazon Textract to automatically label your images.

4. A SageMaker Debugger report for one of your training jobs indicates that you have low GPU
utilization. Which of the following strategies will likely increase your performance while lowering cost?

A. Try a smaller GPU instance type.
B. Try a larger CPU instance type.
C. Try a larger GPU instance type.
D. Try a smaller CPU instance type.

5. A company trains several models—one per ZIP code in the country—and is concerned that
hosting these models on AWS will be very expensive. Which of the following will let the
company easily host these models with minimum cost?

A. Use SageMaker spot endpoints.
B. Use SageMaker on-demand endpoints.
C. Use SageMaker multimodel endpoints.
D. Use one low-cost EC2 instance per model.

6. You are training large PyTorch-based deep learning models using a p3.16xlarge GPU on
SageMaker. As the training time grows, training cost is rapidly starting to become a concern
for leadership. What solution would you recommend to lower cost while still being able to
train deep learning models with the least amount of code change and without impacting
training times?

A. Use TensorFlow 2 instead. TensorFlow models train faster than PyTorch models.
B. Switch to training on EC2 to avoid the cost premiums imposed by managed AWS services.
C. Switch to training on a single p3.2xlarge instead.
D. Use SageMaker spot instances for training.

7. You are serving large PyTorch-based deep learning models using a p3 GPU instance. Since
the serving infrastructure needs to be always on, costs are starting to grow. However, you
have noticed that the GPU utilization during serving is not necessarily high but that serving
on CPUs leads to increased latency and lowered inference throughput. How could you lower
inference costs while still maintaining inference latency and throughput with little to no
code change?

A. Refactor your model and inference code to run using TensorFlow. TF serving is more
performant than TorchServe.
B. Switch to a g4.dn instance that is designed for inference.
C. Switch to an r5 instance that is designed for inference.
D. Use SageMaker Inferentia to deploy models using a custom chip for reduced inference
costs.

8. As your data science team grows, as an administrator you are concerned about rising costs associated with keeping SageMaker notebook instances open and running. Furthermore, you have noticed that while scientists sometimes require GPU-based notebooks, at other times the GPU usage is extremely low as the scientists are primarily exploring data or writing algorithm code. What solution might you consider to lower costs while also maintaining a good customer experience and avoiding long instance startup and switch times?

A. Package IPython and the notebook environment into a container and deploy it on EC2.
Have scientists switch between different EC2 instances based on whether GPU or CPU is
required.
B. Switch to SageMaker Studio, which provides fast-start GPU-and CPU-based notebooks.
Scientists can quickly switch between them in a couple of minutes and optimize cost by
dynamically choosing the right instance for their tasks.
C. Switch to Glue Studio, which provides fast-start GPU- and CPU-based notebooks. Scientists can quickly switch between them in a couple minutes and optimize cost by dynamically choosing the right instance for their tasks.
D. Switch to EMR Studio, which provides fast-start GPU- and CPU-based notebooks.
Scientists can quickly switch between them in a couple minutes and optimize cost by
dynamically choosing the right instance for their tasks.

9. As your data science team grows, as an administrator you are concerned about rising costs
associated with keeping SageMaker notebook instances open and running. To lower costs,
you would like to automatically shut down idle notebooks. What approaches can you use
to do this?

A. Create an AWS Lambda function that calls the StopNotebookInstance API using boto3. Manually call the Lambda function every day at a certain time.
B. Create an AWS Lambda function that calls the StopNotebookInstance API using boto3. Create an EventBridge Rule to trigger the Lambda function at a set time.
C. Specify an idle time and use SageMaker lifecycle config to install a script that calls the Jupyter API periodically to determine if there is any notebook activity in the specified idle time. If not, the script calls the StopNotebookInstance API and stops it.
D. SageMaker notebooks cannot be stopped, only terminated. Save all the notebook data to S3 and terminate the notebook.

10. You have successfully used Amazon Rekognition for building image classification models, but you are getting concerned with the growing costs associated with the fully managed AI service. While you can retrain the model using another AWS infrastructure service, you are still interested in benefiting from pretrained models and fine-tuning them instead of training from scratch. You also want to leverage a managed service for hosting instead of developing your own model inference architecture. What approach could you use to lower cost while still retaining some benefits of managed AWS services?

A. Use Rekognition Custom Labels to still benefit from pretrained models and fine-tune
them. Extract the trained model artifacts to S3 and deploy them yourself using EC2.
B. Use Rekognition Custom Labels to still benefit from pretrained models and fine-tune
them. Extract the trained model artifacts to S3 and deploy them yourself using SageMaker.
C. You cannot train a model on Rekognition and deploy it elsewhere. Use pretrained image
classification models available on TensorFlow Hub for fine-tuning and host them on a
custom hosting service built on EKS.
D. Use SageMaker’s built-i

- Tf-idf is a statistical technique frequently used in Machine Learning domains such as text-summarization and classification. Tf-idf measures the relevance of a word in a document compared to the entire corpus of documents. You have a corpus (D) containing the following documents:
Document 1 (d1) : “A quick brown fox jumps over the lazy dog. What a fox!”
Document 2 (d2) : “A quick brown fox jumps over the lazy fox. What a fox!”
Which of the following statements is correct?

tf is the frequency of any "term" in a given "document". Using this definition, we can compute the following:
tf(“fox”, d1) = 2/12 , as the word "fox" appears twice in the first document which has a total of 12 words
tf(“fox”, d2) = 3/12 , as the word "fox" appears thrice in the second document which has a total of 12 words
An idf is constant per corpus (in this case, the corpus consists of 2 documents) , and accounts for the ratio of documents that include that specific "term". Using this definition, we can compute the following:
idf(“fox”, D) = log(2/2) = 0 , as the word "fox" appears in both the documents in the corpus
Now,
tf-idf(“fox”, d1, D) = tf(“fox”, d1) * idf(“fox”, D) = (2/12) * 0 = 0
tf-idf(“fox”, d2, D) = tf(“fox”, d2) * idf(“fox”, D) = (3/12) * 0 = 0
Using tf-idf, the word “fox” is equally relevant (or just irrelevant!) for both document d1 and document d2

1
https://www.udemy.com/course/aws-machine-learning-practice-exam/learn/quiz/4713424#overview 

2
https://www.udemy.com/course/aws-certified-machine-learning-specialty-full-practice-exams/learn/quiz/4755118#reviews 

2(4)
https://www.udemy.com/course/aws-machine-learning-specialty-certification-test-2021/learn/quiz/5281890#overview 

1
https://www.youtube.com/watch?v=x5mwxrWZulk&ab_channel=DataScienceGarage 
https://www.udemy.com/course/aws-certified-machine-learning-specialty-full-practice-exams/learn/quiz/4755096/results?expanded=813498444#overview
https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf
https://awscertificationpractice.benchprep.com/app/aws-certified-machine-learning-specialty-official-practice-question-set#exams/details/151021 

1
https://www.youtube.com/watch?v=b8HC3LD6sjo&ab_channel=DataScienceGarage 
https://www.udemy.com/course/aws-machine-learning/learn/quiz/4737794#overview
https://www.udemy.com/course/aws-machine-learning/learn/quiz/4731008#notes
https://www.udemy.com/course/aws-machine-learning/learn/quiz/5431588#notes
https://www.udemy.com/course/aws-machine-learning/learn/quiz/5431718#notes
https://www.udemy.com/course/aws-machine-learning/learn/quiz/4731034#overview
https://www.udemy.com/course/aws-machine-learning/learn/quiz/4731030#overview

7
https://www.testpreptraining.com/index.php?route=account/test&test_id=1952 
test 3 https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3508888 
test 4 https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3436056 
test 5 https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3436193
test 6 https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3436401 
test 7 * https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3436794 
test 8 * https://www.testpreptraining.com/index.php?route=account/test/result&quiz_id=3436915 


app - 3
