"""Build the current exchange contract for synthetic test targets."""
from copy import deepcopy

def current_task(value):
    task=deepcopy(value)
    profiles={v['voice']:v['locale'] for v in task.get('voices',[])}
    for item in task.get('items',[]):
        item.setdefault('synthesis',dict(mode='ipa' if item.get('pronunciationId') else 'text',speed='1.0',overrides=[]))
        if not item.get('contentId','').startswith('preview-') and 'identities' not in item:
            item['identities']={voice:dict(text=item['text'],contentType='word' if item.get('pronunciationId') else 'sentence',
                language='en',ipa=item.get('ipa'),locale=profiles[voice],voice=voice,provider='kokoro',model='hexgrad/Kokoro-82M',
                overrides=[],speed='1.0',format='mp3') for voice in item.get('voices',[]) if voice in profiles}
    return task
