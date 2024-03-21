Interaction with a HAPI FHIR database
==============

**Authors:** *JONAS STYLBÄCK, KEER ZHANG*

# About

This project was all about [HAPI FHIR](https://hapifhir.io/), which is a complete implementation of the HL7 FHIR standard for healthcare interoperability. Previously, we'd made a rudementary PPG sensor to measure heartbeats using passive components and a Micro:Bit.

**The task**: Create a patient resource in the HAPI FHIR database. Find a way to upload the heart rate from the PPG sensor to the database, labeled to your patient resource. Next, find a way to extract this data, present key values and draw some graphs.

This would require a set of scripts to upload data using the HL7 FHIR standard to the database, followed by another set of scripts to retrieve data from the database and make some plots. These can be found in the `/code` directory with the final results displayed below:

![](/hapi-fhir-interaction/media/blenotification.png)
![](/hapi-fhir-interaction/media/graphs.png)

# Experience Gained

I learned more about JavaScript and the Plotly library. I was also glad to be working with HAPI FHIR, which have long been a well known quantity in the field of medical informatics.