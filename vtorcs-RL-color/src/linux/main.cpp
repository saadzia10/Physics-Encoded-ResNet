/***************************************************************************

file                 : main.cpp
created              : Sat Mar 18 23:54:30 CET 2000
copyright            : (C) 2000 by Eric Espie
email                : torcs@free.fr
version              : $Id: main.cpp,v 1.14.2.1 2008/11/09 17:50:22 berniw Exp $

***************************************************************************/

/***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************/

#include <stdlib.h>

#include <GL/glut.h>

#include <tgfclient.h>
#include <client.h>

#include "linuxspec.h"
// Raceconfig
#include <raceinit.h>

#include <sys/shm.h>
#define image_width 640
#define image_height 480
#include <iostream>
#include <unistd.h>

// By dosssman
#include <plib/ssg.h> /* To use ssgInit() */
// End By dosssman

extern bool bKeepModules;
// Raceconfig end

// By dosssman
bool runRaceConfigGUI = false;
bool runRaceConfigNoGUI = false;
// End By dosssman

static void
init_args(int argc, char **argv, const char **raceconfig) {
  int		i;
  char	*buf;

  setTextOnly(false);
  setNoisy(false);
  setVersion("2010");
  //dosssman
  setRecordHuman( false);
  init_rec_session_start();
  //end dossman

  i = 1;

  //    for( int asdf = 0; asdf<argc; asdf++)
  //      printf("arg %d: %s\n",asdf,argv[asdf]);

  printf("\n");


  while (i < argc) {
    //    printf("arg %d: %s\n",i,argv[i]);
    if (strncmp(argv[i], "-c", 2) == 0) {
      i++;
      if (i < argc) {
        buf = (char *)malloc(strlen(argv[i]) + 2);
        sprintf(buf, "%s/", argv[i]);
        SetLocalDir(buf);
        free(buf);
        i++;
      }
    } else if (strncmp(argv[i], "-L", 2) == 0) {
      i++;
      if (i < argc) {
        buf = (char *)malloc(strlen(argv[i]) + 2);
        sprintf(buf, "%s/", argv[i]);
        SetLibDir(buf);
        free(buf);
        i++;
      }
    } else if (strncmp(argv[i], "-D", 2) == 0) {
      i++;
      if (i < argc) {
        buf = (char *)malloc(strlen(argv[i]) + 2);
        sprintf(buf, "%s/", argv[i]);
        SetDataDir(buf);
        free(buf);
        i++;
      }
    } else if (strncmp(argv[i], "-s", 2) == 0) {
      i++;
      SetSingleTextureMode ();
    } else if (strncmp(argv[i], "-t", 2) == 0) {
      i++;
      if (i < argc) {
        long int t;
        sscanf(argv[i],"%ld",&t);
        setTimeout(t);
        printf("UDP Timeout set to %ld 10E-6 seconds.\n",t);
        i++;
      }
    } else if (strncmp(argv[i], "-nodamage", 9) == 0) {
      i++;
      setDamageLimit(false);
      printf("Car damages disabled!\n");
    } else if (strncmp(argv[i], "-nofuel", 7) == 0) {
      i++;
      setFuelConsumption(false);
      printf("Fuel consumption disabled!\n");
    } else if (strncmp(argv[i], "-noisy", 6) == 0) {
      i++;
      setNoisy(true);
      printf("Noisy Sensors!\n");
    } else if (strncmp(argv[i], "-ver", 4) == 0) {
      i++;
      if (i < argc) {
        setVersion(argv[i]);
        printf("Set version: \"%s\"\n",getVersion());
        i++;
      }
    } else if (strncmp(argv[i], "-nolaptime", 10) == 0) {
      i++;
      setLaptimeLimit(false);
      printf("Laptime limit disabled!\n");
    } else if (strncmp(argv[i], "-T", 2) == 0) {
      i++;
      setTextOnly(true);
      printf("Text Version!\n");
      // GIUSE - UDP PORT AS ARGUMENT
    } else if (strncmp(argv[i], "-p", 2) == 0) {
      i++;
      setUDPListenPort(atoi(argv[i]));
      i++;
      printf("UDP Listen Port set to %d!\n", getUDPListenPort());
      // GIUSE - VISION HERE! ACTIVATE IMAGE GENERATION (and send them to clients if specified in the car/server)
    } else if (strncmp(argv[i], "-vision", 7) == 0) {
      i++;
      setVision(true);
      printf("Image generation is ON!\n");
      // GIUSE - FASTER THEN RUNTIME ACTIVATION FOR NON-TEXTUAL COMPUTATION
    } else if (strncmp(argv[i], "-a", 2) == 0) {
      i++;
      printf("Speed set to %dx realtime!\n", atoi(argv[i]));
      setSpeedMult((double) (1.0 / (double) atoi(argv[i])));
      i++;

      #ifndef FREEGLUT
    } else if (strncmp(argv[i], "-m", 2) == 0) {
      i++;
      GfuiMouseSetHWPresent(); /* allow the hardware cursor */
      #endif
    }
    // dosssman
    // directly start race from raceconfig file through command line
    else if(strncmp(argv[i], "-raceconfig", 11) == 0) {
      // GfOut( "\n##### DEBUG: race config specified. #####\n");
      i++;
      *raceconfig = "";

      if(i < argc) {
        *raceconfig = argv[i];
        // GfOut( "\n##### DEBUG: race config file: ");
        // GfOut( *raceconfig);
        // GfOut( " \n");
        runRaceConfigGUI = true;
        // TODO Tweak for RACE RESTART even after laps all ended
        // setTextOnly( true);
        i++;
      }

      if((strlen(*raceconfig) == 0) || (strstr(*raceconfig, ".xml") == 0)) {
        printf("Please specify a race configuration xml when using -r\n");
        exit(1);
      }
    }
    else if(strncmp(argv[i], "-runconsole", 11) == 0) {
      // GfOut( "\n##### DEBUG: race config specified. #####\n");
      i++;
      *raceconfig = "";

      if(i < argc) {
        *raceconfig = argv[i];
        // GfOut( "\n##### DEBUG: race config file: ");
        // GfOut( *raceconfig);
        // GfOut( " \n");
        runRaceConfigNoGUI = true;
        i++;
      }

      if((strlen(*raceconfig) == 0) || (strstr(*raceconfig, ".xml") == 0)) {
        printf("Please specify a race configuration xml when using -r\n");
        exit(1);
      }
    }
    else if(strncmp(argv[i], "-rechum", 7) == 0) {
      // GfOut( "\n##### DEBUG: Recording human specified #####\n");
      printf( "Recording first player 's data");
      setRecordHuman( true);

      i++;
      // TODO Implement error managment
      setRecCarIndex( atoi(argv[i]));
      i++;
    }
    else if( strncmp(argv[i], "-rectimesteplim", 15) == 0) {
      if( !getRecordHuman()) {
        printf( "Timestep limit specified but recording seems to be disabled\n");
      }

      i++;
      if( i < argc) {
        // TODO: Check if integer maybe
        setRecTimestepLimit( atoi( argv[i]));
        i++;
      }
    }
    else if( strncmp(argv[i], "-recepisodelim", 14) == 0) {
      if( !getRecordHuman()) {
        printf( "Episode limit specified but recording seems to be disabled\n");
      }

      i++;
      if( i < argc) {
        // TODO: Check if integer maybe
        setRecEpisodeLimit( atoi( argv[i]));
        i++;
      }
    }
    // end dosssman
    else {
      i++;		/* ignore bad args */
    }
  }
  #ifdef FREEGLUT
  GfuiMouseSetHWPresent(); /* allow the hardware cursor (freeglut pb ?) */
  #endif
}

/*
* Function
*	main
*
* Description
*	LINUX entry point of TORCS
*
* Parameters
*
*
* Return
*
*
* Remarks
*
*/
int
main(int argc, char *argv[]) {
  // dosssman
  // enable raceconfig detection
  const char *raceconfig = "";
  init_args(argc, argv, &raceconfig);
  // end dosssman

  LinuxSpecInit();		/* init specific linux functions */

  // dosssman
  if( runRaceConfigGUI) {
    //Directly init race
    GfScrInit(argc, argv);	/* init screen */

    ssgInit();
    GfInitClient();

    ReGuiWithoutSelect(raceconfig);

  } else if( runRaceConfigNoGUI) {
    ReRunRaceOnConsole(raceconfig);
  } else {
  // End dosssman

    if (getTextOnly()==false)
    GfScrInit(argc, argv);	/* init screen */

    TorcsEntry();		/* launch TORCS */

    if (getTextOnly()==false)
      glutMainLoop();		/* event loop of glut */
  }

  return 0;			/* just for the compiler, never reached */
}
