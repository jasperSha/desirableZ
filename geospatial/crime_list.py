#!/usr/bin/env python3



misc_crime = [
    'BOMB SCARE',
    'BRANDISH WEAPONTRESPASSING',
    'BRIBERY',
    'Counterfeit Suspect There Now',
    'VIOLATION OF RESTRAINING ORDER',
    'Violation of Restraining Order In Progress',
    'Violation of Restraining Order Report',
    'Trespassing',
    'Urinating/Defecating in Public',
    'Stalking Suspect Just Left',
    'STALKING',
    'SUICIDE AND ATTEMPT',
    'RECKLESS DRIVING',
    'DISORDERLY CONDUCT',
    'DISTURBING THE PEACE',
    'DRUGS, TO A MINOR',
    'DRUNK / ALCOHOL / DRUGS',
    'DRUNK DRIVING VEHICLE / BOATWEAPON LAWS',
    'DUI ARREST',
    'Disturbance of the Peace',
    'Disturbance of the Peace Report',
    'Drunk Driving Investigation',
    'Elder Abuse',
    'FAILURE TO DISPERSE',
    'FEDERAL OFFENSES WITH MONEY',
    'FELONIES MISCELLANEOUS',
    'FIREARMS TEMPORARY RESTRAINING ORDER (TEMP FIREARMS RO)',   
    'FRAUD AND NSF CHECKS',
    'Fight',
    'Hit and Run Felony Investigation',
    'INCITING A RIOT',
    'INDECENT EXPOSURE',
    'Illegal Weapon',
    'Indecent Exposure Just Occurred',
    'Indecent Exposure Report',
    'Bomb Threat',
    'MISDEMEANORS MISCELLANEOUS',
    'NARCOTICS',
    'OFFENSES AGAINST FAMILY',
    'OTHER MISCELLANEOUS CRIME',
    'PERSONS MISSING',
    'Panhandling',
    'Person with a Gun'
     
]



violent_crime = [
    'OTHER ASSAULT',
    'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS',
    'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE',
    'Attempt Homicde',
    'AGGRAVATED ASSAULT',
    'ASSAULT - AGGRAVATED',
    'DISCHARGE FIREARMS/SHOTS FIRED',
    'ASSAULT - SIMPLE',
    'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER',
    'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
    'ASSAULTS (PRIOR TO SEPT 2018)',
    'Assault Just Occurred',
    'Assault Now',
    'Assault ReportProstitution',
    'Report of Shots Fired',
    'Assault w/Deadly Weapon Report',
    'PERSONS DEAD',
    'Assault w/Deadly Wepaon Just Occurred',
    'Assault w/Deadly Wepaon Now',
    'BATTERY - SIMPLE ASSAULT',
    'BATTERY ON A FIREFIGHTER',
    'SHOTS FIRED AT INHABITED DWELLING',
    'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT',
    'WEAPONS POSSESSION/BOMBING',
    'BATTERY POLICE (SIMPLE)',
    'BATTERY WITH SEXUAL CONTACT',
    'Battery Now',
    'CRIMINAL HOMICIDE',
    'KIDNAPPING',
    'KIDNAPPING - GRAND ATTEMPT',
    'Kidnap In Progress',
    'Kidnap Just Occurred',
    'Kidnap Report',
    'FORCIBLE RAPE',
    'Domestic Violence Just Occurred',
    'Domestic Violence Report',
    'Homicide',
    'INTIMATE PARTNER - AGGRAVATED ASSAULT',
    'INTIMATE PARTNER - SIMPLE ASSAULT',
    'LYNCHING',
    'LYNCHING - ATTEMPTED',
    'MANSLAUGHTER, NEGLIGENT',
    'NON-AGGRAVATED ASSAULTS',
    'RAPE, ATTEMPTED',
    'RAPE, FORCIBLE',
    'Rape Just Occurred',
    'Rape Now',
    'Rape Report',
    'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ',
    'SEXUAL PENETRATION W/FOREIGN OBJECT',
    'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH',
    'Sexual Assault'
]


property_crime = [
    'Stolen Vehicle',
    'ROBBERY',
    'Receiving Stolen Property Report',
    'LARCENY THEFT',
    'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)',
    'EXTORTION',
    'ARSON',
    'Arson',
    'Wanted Vehicle',
    'ATTEMPTED ROBBERY',
    'Armed Robbery In Progress',
    'Armed Robbery Just Occurred',
    'Armed Robbery Report',
    'Armed Robbery/Auto Theft In Progress',
    'Armed Robbery/Auto Theft Just Occurred',
    'Armed Robbery/Auto Theft Report',
    'Attempt Armed Robbery Just Occurred',
    'Attempt Armed Robbery Report',
    'Attempt Auto Theft Just Occurred',
    'Attempt Auto Theft Report',
    'Attempt Burglary Just Occurred',
    'Attempt Burglary Report',
    'Attempt Strongarm Robbery Just Occurred',
    'Attempt Strongarm Robbery Report',
    'Auto Burglary Just Occurred',
    'Auto Burglary Now',
    'BIKE - STOLEN',
    'BOAT - STOLEN',
    'BUNCO, ATTEMPT',
    'BUNCO, GRAND THEFT',
    'BUNCO, PETTY THEFT',
    'CREDIT CARDS, FRAUD USE ($950.01 & OVER)',
    'BURGLARY',
    'BURGLARY - COMMERCIAL BUILDING',
    'BURGLARY - CONSTRUCTION SITE',
    'BURGLARY - FROM A MOTOR VEHICLE',
    'BURGLARY - RESIDENTIAL (ACCESSED GARAGE ONLY)',
    'BURGLARY - RESIDENTIAL (COMMON AREA)',
    'BURGLARY - RESIDENTIAL (HOME OCCUPIED)',
    'BURGLARY - RESIDENTIAL (NO ONE HOME)',
    'BURGLARY FROM VEHICLE',
    'BURGLARY FROM VEHICLE, ATTEMPTED',
    'BURGLARY, ATTEMPTED',
    'Burglary Investigation/Walk Through',
    'Burglary Just Occurred',
    'Burglary Now',
    'Burglary Report',
    'DISHONEST EMPLOYEE - GRAND THEFT',
    'DISHONEST EMPLOYEE ATTEMPTED THEFT',
    'GRAND THEFT / AUTO REPAIR',
    'GRAND THEFT / INSURANCE FRAUD',
    'GRAND THEFT AUTO',
    'Grand Theft Auto In Progress',
    'Grand Theft Auto Just Occurred',
    'Grand Theft Auto Report',
    'Grand Theft Just Occurred',
    'Grand Theft Now',
    'Grand Theft Report',
    'Petty Theft Report',
    'Strongarm Robbery Just Occurred',
    'Strongarm Robbery Now',
    'Strongarm Robbery Report',
    'THEFT - GRAND',
    'THEFT - GRAND (FROM VEHICLE)',
    'THEFT - PETTY',
    'THEFT - PETTY (FROM VEHICLE)',
    'THEFT FROM MOTOR VEHICLE - ATTEMPT',
    'THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)',
    'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)',
    'THEFT FROM PERSON - ATTEMPT',
    'THEFT PLAIN - ATTEMPT',
    'THEFT PLAIN - PETTY ($950 & UNDER)',
    'THEFT, COIN MACHINE - GRAND ($950.01 & OVER)',
    'THEFT, PERSON',
    'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD',
    'TILL TAP - GRAND THEFT ($950.01 & OVER)',
    'TILL TAP - PETTY ($950 & UNDER)',
    'LOJACK Hit',
    'MOTOR VEHICLE THEFT',
    'PICKPOCKET',
    'PICKPOCKET, ATTEMPT',
    'PURSE SNATCHING',
    'PURSE SNATCHING - ATTEMPT',
    'SHOPLIFT ROBBERY',
    'SHOPLIFTING - ATTEMPT',
    'SHOPLIFTING - PETTY THEFT ($950 & UNDER)',
    'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)',
    'VEHICLE - ATTEMPT STOLEN',
    'VEHICLE - STOLENVIOLATION OF COURT ORDER',
    'VANDALISM',
    'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)',
    'VANDALISM - MISDEAMEANOR ($399 OR UNDER)'
]


deviant_crime = [
    'BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM',
    'CHILD ABANDONMENT',
    'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
    'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT',
    'CHILD NEGLECT (SEE 300 W.I.C.)',
    'CHILD PORNOGRAPHY',
    'CHILD STEALING',
    'CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)',
    'CRUELTY TO ANIMALS',
    'Child Abuse',
    'Child Endangerment',
    'Child Molestation',
    'Child Stealing',
    'SEX OFFENSES FELONIES',
    'SEX OFFENSES MISDEMEANORS',
    'LEWD CONDUCT',
    'LEWD/LASCIVIOUS ACTS WITH CHILD',
    'INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)'
 
]



















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


