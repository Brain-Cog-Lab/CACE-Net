import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, num_layers):
        super(RNNEncoder, self).__init__()

        self.audio_rnn = nn.LSTM(audio_dim, audio_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, video_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.visual_projecter = nn.Linear(video_dim, audio_dim)
        self.audio_projecter = nn.Linear(audio_dim, audio_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feature, guide_audio_feature, visual_feature):
        audio_output = self.relu(self.audio_projecter(self.audio_rnn(audio_feature)[0]))
        guide_audio_output = self.relu(self.audio_projecter(self.audio_rnn(guide_audio_feature)[0]))
        video_output = self.relu(self.visual_projecter(self.visual_rnn(visual_feature)[0]))
        return audio_output, guide_audio_output, video_output


class Encoder(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""EncoderLayer, which is borrowed from CMRAN.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(Module):
    r"""DecoderLayer, which is borrowed from CMRAN.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)



class Only_Audio_Guided_Attention(nn.Module):
    r"""
    This implementation is slightly different from what we described in the paper, which we later found it to be more efficient.
    
    """
   
    def __init__(self, beta):
        super(Only_Audio_Guided_Attention, self).__init__()

        self.beta = beta
        self.relu = nn.ReLU()
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.hidden_dim = 256
        # channel attention
        self.affine_video_1 = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.affine_audio_1 = nn.Linear(self.audio_input_dim, self.video_input_dim)
        self.affine_bottleneck = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_v_c_att = nn.Linear(self.hidden_dim, self.video_input_dim)
        # spatial attention
        self.affine_video_2 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_audio_2 = nn.Linear(self.audio_input_dim, self.hidden_dim)
        self.affine_v_s_att = nn.Linear(self.hidden_dim, 1)


        self.latent_dim = 4
        self.video_query = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_key = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_value = nn.Linear(self.video_input_dim, self.video_input_dim)

        self.affine_video_ave = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_video_3 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.ave_bottleneck = nn.Linear(512, 256)
        self.ave_v_att = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.video_input_dim)


    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        raw_audio_feature = audio
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # ============================== Self Attention =======================================
        #visual_feature = c_att_visual_feat
        video_query_feature = self.video_query(visual_feature).reshape(batch * t_size, h * w, -1)  # [B, h*w, C]
        video_key_feature = self.video_key(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value(visual_feature).reshape(batch * t_size, h * w, -1)
        output = torch.matmul(attention, video_value_feature)
        output = self.norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output))
        #c_att_visual_feat = output
        visual_feature = output
        # ============================== Video Spatial Attention ====================================
        video_average = visual_feature.sum(dim=1)/(h*w)
        video_average = video_average.reshape(batch*t_size, v_dim)
        video_average = self.relu(self.affine_video_ave(video_average)).unsqueeze(-2)
        self_video_att_feat = visual_feature.reshape(batch*t_size, -1, v_dim)
        self_video_att_query = self.relu(self.affine_video_3(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query)).transpose(2,1))
        self_att_feat = torch.bmm(self_spatial_att_maps, visual_feature).squeeze().reshape(batch, t_size, v_dim)


        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch*t_size, h*w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))


        # ==============================Audio Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch*t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        #audio_query_2 = self.relu(self.audio_ff(audio_query_1))
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        c_s_att_visual_feat = c_s_att_visual_feat + self.beta*self_att_feat.sigmoid()*c_s_att_visual_feat


        return c_s_att_visual_feat, raw_audio_feature



class Co_Guided_Attention(nn.Module):
    def __init__(self, psai, beta, video_input_dim, audio_input_dim):
        super(Co_Guided_Attention, self).__init__()

        self.video_input_dim = video_input_dim  # 512
        self.audio_input_dim = audio_input_dim  # 128
        self.hidden_dim = 256  # 256
        self.beta = beta
        self.psai =psai
        self.relu = nn.ReLU()

        # channel attention
        self.affine_video_1 = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.affine_audio_1 = nn.Linear(self.audio_input_dim, self.video_input_dim)
        self.affine_bottleneck = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_v_c_att = nn.Linear(self.hidden_dim, self.video_input_dim)
        # spatial attention
        self.affine_video_2 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_audio_2 = nn.Linear(self.audio_input_dim, self.hidden_dim)
        self.affine_v_s_att = nn.Linear(self.hidden_dim, 1)

        # video-guided audio attention
        self.latent_dim = 4
        self.video_query = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_key = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_value = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.v_norm = nn.LayerNorm(self.video_input_dim)

        self.latent_dim = 4
        self.video_query2 = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_key2 = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_value2 = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.v_norm2 = nn.LayerNorm(self.video_input_dim)

        self.affine_video_ave = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_video_3 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.ave_bottleneck = nn.Linear(512, 256)
        self.ave_v_att = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)

        self.v_query = nn.Linear(self.video_input_dim, self.audio_input_dim)
        self.vaudio_key = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.vaudio_value = nn.Linear(self.audio_input_dim, self.audio_input_dim)

        self.audio_query = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.audio_key = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.audio_value = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.a_norm = nn.LayerNorm(self.audio_input_dim)


    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]; [batch, 10, 128]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        raw_audio_feature = audio
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # ------------------- Step1: Audio-Guide, CMGA�ķ���, ����Ӿ����� ------------------------#
        # ============================== Self Attention =======================================
        #visual_feature = c_att_visual_feat
        video_query_feature = self.video_query(visual_feature).reshape(batch * t_size, h * w, -1)  # [B, h*w, C]
        video_key_feature = self.video_key(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value(visual_feature).reshape(batch * t_size, h * w, -1)
        output = torch.matmul(attention, video_value_feature)
        output = self.v_norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output))

        video_query_feature = self.video_query2(visual_feature).reshape(batch * t_size, h * w, -1)  # [B, h*w, C]
        video_key_feature = self.video_key2(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value2(visual_feature).reshape(batch * t_size, h * w, -1)
        output2 = torch.matmul(attention, video_value_feature)
        output2 = self.v_norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output2))

        visual_feature = output
        visual_feature2 = output2

        # ============================== Video Spatial Attention ====================================
        video_average = visual_feature.sum(dim=1)/(h*w)  # visual_feature shape [B, T, C]
        video_average = video_average.reshape(batch*t_size, v_dim)
        video_average = self.relu(self.affine_video_ave(video_average)).unsqueeze(-2)
        self_video_att_feat = visual_feature.reshape(batch*t_size, -1, v_dim)
        self_video_att_query = self.relu(self.affine_video_3(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query)).transpose(2,1))
        self_att_feat = torch.bmm(self_spatial_att_maps, visual_feature).squeeze().reshape(batch, t_size, v_dim)

        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch*t_size, h*w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch*t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        guide_visual_feature = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)  # CMRAN

        guide_visual_feature = guide_visual_feature + self.beta*self_att_feat.sigmoid()*guide_visual_feature  # [B, T, C]  # CMBS

        # ------------------- Step2: Viusal-Guide, �����Ƶ���� ------------------------#
        # ============================== Audio Self Attention ====================================
        audio_query_feature = self.audio_query(raw_audio_feature)  # [B, T, C]
        audio_key_feature = self.audio_key(raw_audio_feature).permute(0, 2, 1)  # [B, C, T]
        energy = torch.bmm(audio_query_feature, audio_key_feature)
        attention = self.softmax(energy)
        audio_value_feature = self.audio_value(raw_audio_feature)
        output = torch.matmul(attention, audio_value_feature)  # [B, T, C]
        tmp_guide_audio_feature = self.a_norm(raw_audio_feature + self.dropout(output))  # [B, T, C]
        # tmp_guide_audio_feature = raw_audio_feature

        # ============================== Visual-Guide Attention ====================================
        video_query = self.v_query(visual_feature2.sum(dim=1).reshape(batch, t_size, v_dim)/(h*w))  # [B, T, C]
        audio_key = self.vaudio_key(tmp_guide_audio_feature).permute(0, 2, 1)
        energy = torch.bmm(video_query, audio_key)
        attention = self.softmax(energy)  # [B, T, T]
        audio_value = self.vaudio_value(tmp_guide_audio_feature)  # [B, T, C]
        output = torch.matmul(attention, audio_value)  # [B, T, C]
        # guide_audio_feature = self.a_norm(tmp_guide_audio_feature + self.dropout(output))  # [B, T, C]
        guide_audio_feature = output  # [B, T, C]

        # ----------�������ǻ�����ĸ�����---------------#
        # raw_visual_feature, guide_visual_feature, raw_audio_feature, guide_audio_feature
        # audio_score, guide_audio_score, visual_score = self.audio_visual_rnn_layer(raw_audio_feature, guide_audio_feature, guide_visual_feature)

        # cos_sim = F.cosine_similarity(visual_score, audio_score, dim=-1)
        # guide_cos_sim = torch.abs(F.cosine_similarity(visual_score, guide_audio_score, dim=-1))

        # �Ƚϵ�һ����������ȡ���ֵ������
        # _, indices = torch.max(torch.stack((cos_sim, guide_cos_sim)), 0)
        # ���������ӵڶ�����������ѡԪ��
        # audio_feature = torch.where(indices.unsqueeze(-1) == 0, audio_score, guide_audio_score)

        raw_audio_feature = raw_audio_feature + self.psai * guide_audio_feature.sigmoid() * raw_audio_feature
        # raw_audio_feature = guide_audio_feature
        # audio_feature = raw_audio_feature + guide_cos_sim.unsqueeze(-1) * guide_audio_feature * 0.1

        return guide_visual_feature, raw_audio_feature



class Only_Visual_Guided_Attention(nn.Module):
    def __init__(self, psai):
        super(Only_Visual_Guided_Attention, self).__init__()

        self.video_input_dim = 512
        self.audio_input_dim = 128
        self.hidden_dim = 256
        self.beta = 0.4
        self.psai = psai
        self.relu = nn.ReLU()

        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.latent_dim = 4
        self.video_query = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_key = nn.Linear(self.video_input_dim, self.video_input_dim//self.latent_dim)
        self.video_value = nn.Linear(self.video_input_dim, self.video_input_dim)
        self.v_norm = nn.LayerNorm(self.video_input_dim)

        self.affine_video_ave = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.affine_video_3 = nn.Linear(self.video_input_dim, self.hidden_dim)
        self.ave_bottleneck = nn.Linear(512, 256)
        self.ave_v_att = nn.Linear(self.hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)

        self.v_query = nn.Linear(self.video_input_dim, self.audio_input_dim)
        self.vaudio_key = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.vaudio_value = nn.Linear(self.audio_input_dim, self.audio_input_dim)

        self.audio_query = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.audio_key = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.audio_value = nn.Linear(self.audio_input_dim, self.audio_input_dim)
        self.a_norm = nn.LayerNorm(self.audio_input_dim)


    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]; [batch, 10, 128]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        raw_audio_feature = audio
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # ------------------- Step1: Audio-Guide, CMGA�ķ���, ����Ӿ����� ------------------------#
        # ============================== Self Attention =======================================
        #visual_feature = c_att_visual_feat
        video_query_feature = self.video_query(visual_feature).reshape(batch * t_size, h * w, -1)  # [B, h*w, C]
        video_key_feature = self.video_key(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value(visual_feature).reshape(batch * t_size, h * w, -1)
        output = torch.matmul(attention, video_value_feature)
        output = self.v_norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output))

        visual_feature = output

        guide_visual_feature = raw_visual_feature.sum(dim=2).reshape(batch, t_size, v_dim)/(h*w)

        # ------------------- Step2: Viusal-Guide, �����Ƶ���� ------------------------#
        # ============================== Audio Self Attention ====================================
        audio_query_feature = self.audio_query(raw_audio_feature)  # [B, T, C]
        audio_key_feature = self.audio_key(raw_audio_feature).permute(0, 2, 1)  # [B, C, T]
        energy = torch.bmm(audio_query_feature, audio_key_feature)
        attention = self.softmax(energy)
        audio_value_feature = self.audio_value(raw_audio_feature)
        output = torch.matmul(attention, audio_value_feature)  # [B, T, C]
        tmp_guide_audio_feature = self.a_norm(raw_audio_feature + self.dropout(output))  # [B, T, C]
        # tmp_guide_audio_feature = raw_audio_feature

        # ============================== Visual-Guide Attention ====================================
        video_query = self.v_query(visual_feature.sum(dim=1).reshape(batch, t_size, v_dim)/(h*w))  # [B, T, C]
        audio_key = self.vaudio_key(tmp_guide_audio_feature).permute(0, 2, 1)
        energy = torch.bmm(video_query, audio_key)
        attention = self.softmax(energy)  # [B, T, T]
        audio_value = self.vaudio_value(tmp_guide_audio_feature)  # [B, T, C]
        output = torch.matmul(attention, audio_value)  # [B, T, C]
        # guide_audio_feature = self.a_norm(tmp_guide_audio_feature + self.dropout(output))  # [B, T, C]
        guide_audio_feature = output  # [B, T, C]

        # ----------�������ǻ�����ĸ�����---------------#
        # raw_visual_feature, guide_visual_feature, raw_audio_feature, guide_audio_feature
        # audio_score, guide_audio_score, visual_score = self.audio_visual_rnn_layer(raw_audio_feature, guide_audio_feature, guide_visual_feature)

        # cos_sim = F.cosine_similarity(visual_score, audio_score, dim=-1)
        # guide_cos_sim = torch.abs(F.cosine_similarity(visual_score, guide_audio_score, dim=-1))

        # �Ƚϵ�һ����������ȡ���ֵ������
        # _, indices = torch.max(torch.stack((cos_sim, guide_cos_sim)), 0)
        # ���������ӵڶ�����������ѡԪ��
        # audio_feature = torch.where(indices.unsqueeze(-1) == 0, audio_score, guide_audio_score)

        raw_audio_feature = raw_audio_feature + self.psai * guide_audio_feature.sigmoid() * raw_audio_feature
        # raw_audio_feature = guide_audio_feature
        # audio_feature = raw_audio_feature + guide_cos_sim.unsqueeze(-1) * guide_audio_feature * 0.1

        return guide_visual_feature, raw_audio_feature