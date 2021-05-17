import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList, Stack, Box } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import NextLink from 'next/link'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements, video_url } from 'data'


const Index = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Video (disabled for now) */}
    { /* 
    <Container w="90vw" h="50.6vw" maxW="700px" maxH="393px" mb="3rem">
      <iframe
        width="100%" height="100%"
        src={video_url}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container>
    */ }

    {/* Main */}
    <Container w="100%" maxW="44rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">{abstract}</Text>

      {/* Example */}
      <Heading fontSize="2xl" pb="1rem" mb="1rem">Examples</Heading>
      <Stack direction="row" justifyContent="center">
        <video loop autoPlay muted>
          <source src={`${process.env.BASE_PATH || ""}/images/54.mp4`} type="video/mp4" />
        </video>
        <video loop autoPlay muted>
          <source src={`${process.env.BASE_PATH || ""}/images/63.mp4`} type="video/mp4" />
        </video>
        <video loop autoPlay muted>
          <source src={`${process.env.BASE_PATH || ""}/images/78.mp4`} type="video/mp4" />
        </video>
        <video loop autoPlay muted>
          <source src={`${process.env.BASE_PATH || ""}/images/83.mp4`} type="video/mp4" />
        </video>
      </Stack>
      <Text align="center" pt="0.5rem" pb="0.5rem" mb="1rem" fontSize="small">Examples of interpolations in GAN latent space</Text>

      <img src={`${process.env.BASE_PATH || ""}/images/example.png`} />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">Examples of segmentations produced by the final segmentation model, which was trained entirely on GAN-generated images</Text>

      {/* Another Section */}
      {/* <Heading fontSize="2xl" pt="2rem" pb="2rem" id="dataset">Another Section</Heading>
      <Text >
        Here we have...
      </Text> */}

      {/* Citation */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
        &#125;
      </Code>

      {/* Acknowledgements */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        {acknowledgements}
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index
